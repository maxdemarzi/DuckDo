//===----------------------------------------------------------------------===//
// Phase 5-6: the causal foundation model runtime.
//
// Do-PFN reads a labelled context and a set of query rows and returns logits
// over a 100-bucket bar distribution of the interventional outcome. Column 0 of
// its input is the treatment, so CATE is the difference between the
// distribution means under do(T = 1) and do(T = 0) - one forward pass each.
//
// The graphs are traced at fixed context lengths (see scripts/export/
// export_dopfn.py for why), so the runtime picks the largest rung that fits and
// subsamples the context down to it, stratified by arm and seeded.
//===----------------------------------------------------------------------===//
#include "duckdo/runtime.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/functions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>

#ifdef DUCKDO_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace duckdb {
namespace duckdo {

const vector<ModelInfo> &ModelCatalog() {
	static const vector<ModelInfo> catalog = [] {
		vector<ModelInfo> models;
		ModelInfo causalpfn;
		causalpfn.kind = ModelKind::CAUSALPFN;
		causalpfn.id = "causalpfn";
		causalpfn.setting = "backdoor (ignorability)";
		causalpfn.license = "Apache-2.0";
		causalpfn.source = "https://github.com/vdblm/CausalPFN";
		causalpfn.commercial = true;
		causalpfn.attribution_required = false;
		causalpfn.max_covariates = 99;
		causalpfn.num_buckets = 0;
		causalpfn.max_context = 4096;
		causalpfn.weights_file = "causalpfn.weights.bin";
		causalpfn.graph_pattern = "causalpfn.onnx";
		causalpfn.manifest_file = "causalpfn.manifest.json";
		models.push_back(std::move(causalpfn));

		ModelInfo dopfn;
		dopfn.kind = ModelKind::DOPFN;
		dopfn.id = "do_pfn";
		dopfn.setting = "non-identifiable prior";
		dopfn.license = "CC BY 4.0";
		dopfn.source = "https://github.com/jr2021/Do-PFN";
		dopfn.commercial = true;
		dopfn.attribution_required = true;
		dopfn.max_covariates = 5;
		dopfn.num_buckets = 100;
		dopfn.context_ladder = {128, 512, 1024, 2048};
		dopfn.weights_file = "dopfn.weights.bin";
		dopfn.graph_pattern = "dopfn_ctx%llu.onnx";
		dopfn.manifest_file = "dopfn.manifest.json";
		models.push_back(std::move(dopfn));
		return models;
	}();
	return catalog;
}

const ModelInfo *FindModel(const string &id) {
	for (auto &model : ModelCatalog()) {
		if (StringUtil::CIEquals(model.id, id)) {
			return &model;
		}
	}
	return nullptr;
}

string KnownModels() {
	string out;
	for (auto &model : ModelCatalog()) {
		if (!out.empty()) {
			out += ", ";
		}
		out += model.id;
	}
	return out;
}

bool OnnxAvailable() {
#ifdef DUCKDO_WITH_ONNX
	return true;
#else
	return false;
#endif
}

vector<string> AvailableDevices() {
	vector<string> devices;
#ifdef DUCKDO_WITH_ONNX
	devices.push_back("cpu");
#endif
	return devices;
}

string ModelDir(ClientContext &context) {
	const string configured = GetSettingString(context, "duckdo_model_dir", "");
	if (!configured.empty()) {
		return configured;
	}
	const char *home = std::getenv("USERPROFILE");
	if (!home) {
		home = std::getenv("HOME");
	}
	if (!home) {
		return ".duckdo";
	}
	return string(home) + "/.cache/duckdo";
}

static bool FileExists(const string &path) {
	std::ifstream probe(path.c_str(), std::ios::binary);
	return probe.good();
}

static string GraphPath(const string &dir, const ModelInfo &model, idx_t rung) {
	if (model.context_ladder.empty()) {
		return dir + "/" + model.graph_pattern;
	}
	return dir + "/" + StringUtil::Format(model.graph_pattern, static_cast<unsigned long long>(rung));
}

bool ModelArtifactsPresent(ClientContext &context, const ModelInfo &model, string &missing) {
	const string dir = ModelDir(context);
	if (!FileExists(dir + "/" + model.weights_file)) {
		missing = model.weights_file;
		return false;
	}
	if (model.context_ladder.empty()) {
		if (!FileExists(dir + "/" + model.graph_pattern)) {
			missing = model.graph_pattern;
			return false;
		}
	} else {
		for (auto rung : model.context_ladder) {
			const string path = GraphPath(dir, model, rung);
			if (!FileExists(path)) {
				missing = StringUtil::Format(model.graph_pattern, static_cast<unsigned long long>(rung));
				return false;
			}
		}
	}
	if (!model.manifest_file.empty() && !FileExists(dir + "/" + model.manifest_file)) {
		missing = model.manifest_file;
		return false;
	}
	missing.clear();
	return true;
}

// --- inference --------------------------------------------------------------

namespace {

//! Bar-distribution bucket centres, read from the manifest the exporter wrote.
struct BarDistribution {
	vector<double> centres;
	bool loaded = false;
};

//! Minimal extraction of the borders array from the manifest, so the runtime
//! does not need a JSON dependency for one well-known field.
BarDistribution LoadBorders(const string &dir, const string &manifest) {
	BarDistribution out;
	std::ifstream file((dir + "/" + manifest).c_str());
	if (!file.good()) {
		return out;
	}
	string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	const string key = "\"bar_distribution_borders\"";
	auto pos = content.find(key);
	if (pos == string::npos) {
		return out;
	}
	pos = content.find('[', pos);
	const auto end = content.find(']', pos);
	if (pos == string::npos || end == string::npos) {
		return out;
	}
	vector<double> borders;
	string token;
	for (auto i = pos + 1; i < end; i++) {
		const char c = content[i];
		if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
			if (!token.empty()) {
				borders.push_back(std::strtod(token.c_str(), nullptr));
				token.clear();
			}
		} else {
			token += c;
		}
	}
	if (!token.empty()) {
		borders.push_back(std::strtod(token.c_str(), nullptr));
	}
	if (borders.size() < 2) {
		return out;
	}
	out.centres.reserve(borders.size() - 1);
	for (idx_t i = 0; i + 1 < borders.size(); i++) {
		out.centres.push_back(0.5 * (borders[i] + borders[i + 1]));
	}
	out.loaded = true;
	return out;
}

//! Rank covariates by |correlation| with the outcome and keep the budget the
//! model allows. Do-PFN accepts six columns total and the first is the
//! treatment, so five covariates survive.
vector<idx_t> SelectFeatures(const CausalFrame &frame, idx_t budget) {
	vector<std::pair<double, idx_t>> scored;
	const double y_sd = StdDev(frame.y);
	const double y_mean = Mean(frame.y);
	for (idx_t j = 0; j < frame.X.cols; j++) {
		double cov = 0.0;
		for (idx_t i = 0; i < frame.n; i++) {
			cov += frame.X.At(i, j) * (frame.y[i] - y_mean);
		}
		cov /= static_cast<double>(frame.n);
		// Features are already standardised, so this is the correlation up to
		// the outcome's own scale.
		scored.emplace_back(y_sd > 0 ? std::fabs(cov) / y_sd : 0.0, j);
	}
	std::stable_sort(scored.begin(), scored.end(),
	                 [](const std::pair<double, idx_t> &a, const std::pair<double, idx_t> &b) {
		                 return a.first > b.first;
	                 });
	vector<idx_t> kept;
	for (idx_t i = 0; i < scored.size() && kept.size() < budget; i++) {
		kept.push_back(scored[i].second);
	}
	std::sort(kept.begin(), kept.end());
	return kept;
}

//! Seeded, arm-stratified context sample of exactly `size` rows.
//!
//! With `resample` the draw is taken with replacement. That matters: when the
//! table already fits inside the model's context window, sampling without
//! replacement returns every row no matter the seed, so an ensemble of such
//! draws is identical and measures nothing. A bootstrap of the context is the
//! honest way to ask how much the answer depends on which rows the model saw.
vector<idx_t> SampleContext(const CausalFrame &frame, idx_t size, int64_t seed, bool resample = false) {
	std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0xC0FFEEULL);
	vector<idx_t> treated = frame.ArmRows(1.0);
	vector<idx_t> control = frame.ArmRows(0.0);
	std::shuffle(treated.begin(), treated.end(), rng);
	std::shuffle(control.begin(), control.end(), rng);

	// Keep the arm proportions of the source table.
	idx_t want_treated = static_cast<idx_t>(std::llround(
	    static_cast<double>(size) * static_cast<double>(treated.size()) / static_cast<double>(frame.n)));
	want_treated = std::min(want_treated, treated.size());
	idx_t want_control = size - want_treated;
	if (want_control > control.size()) {
		want_control = control.size();
		want_treated = std::min(size - want_control, treated.size());
	}
	vector<idx_t> rows;
	if (resample) {
		// Draw with replacement within each arm, preserving the arm proportions.
		if (!treated.empty()) {
			std::uniform_int_distribution<idx_t> pick(0, treated.size() - 1);
			for (idx_t i = 0; i < want_treated; i++) {
				rows.push_back(treated[pick(rng)]);
			}
		}
		if (!control.empty()) {
			std::uniform_int_distribution<idx_t> pick(0, control.size() - 1);
			for (idx_t i = 0; i < want_control; i++) {
				rows.push_back(control[pick(rng)]);
			}
		}
	} else {
		rows.insert(rows.end(), treated.begin(), treated.begin() + want_treated);
		rows.insert(rows.end(), control.begin(), control.begin() + want_control);
	}
	// Stable order so a repeat run feeds the model identical context.
	std::sort(rows.begin(), rows.end());
	return rows;
}

} // namespace

#ifndef DUCKDO_WITH_ONNX

CfmResult CfmEstimateCate(ClientContext &, const CausalFrame &, const CausalSpec &, const ModelInfo &model) {
	throw BinderException("duckdo: this build has no ONNX Runtime, so model := '%s' cannot run. Rebuild with "
	                      "-DDUCKDO_ONNXRUNTIME_ROOT=<path to an onnxruntime release>, or use one of the classical "
	                      "estimators (estimator := 'aipw' is the doubly-robust default)",
	                      model.id);
}

#else

namespace {

Ort::Env &OrtEnvironment() {
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "duckdo");
	return env;
}

//! One ORT session per (model, rung), created on first use and kept.
struct SessionCache {
	std::mutex lock;
	std::unordered_map<string, unique_ptr<Ort::Session>> sessions;

	static SessionCache &Get() {
		static SessionCache instance;
		return instance;
	}
};

Ort::Session &AcquireSession(const string &path, idx_t threads) {
	auto &cache = SessionCache::Get();
	std::lock_guard<std::mutex> guard(cache.lock);
	auto entry = cache.sessions.find(path);
	if (entry != cache.sessions.end()) {
		return *entry->second;
	}
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(static_cast<int>(std::max<idx_t>(threads, 1)));
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
	const std::wstring wide(path.begin(), path.end());
	auto session = make_uniq<Ort::Session>(OrtEnvironment(), wide.c_str(), options);
#else
	auto session = make_uniq<Ort::Session>(OrtEnvironment(), path.c_str(), options);
#endif
	auto &ref = *session;
	cache.sessions[path] = std::move(session);
	return ref;
}

} // namespace

namespace {

//! Shared preparation: pick features, sample a context, and report what was done.
struct Prepared {
	vector<idx_t> features;
	vector<idx_t> context_rows;
	vector<string> warnings;
};

Prepared PrepareCommon(const CausalFrame &frame, const CausalSpec &spec, const ModelInfo &model,
                       idx_t context_size, bool resample) {
	Prepared out;
	out.features = SelectFeatures(frame, model.max_covariates);
	if (frame.X.cols > model.max_covariates) {
		out.warnings.push_back(StringUtil::Format(
		    "%s accepts %llu covariates; the %llu most outcome-correlated were kept out of %llu",
		    model.id.c_str(), static_cast<unsigned long long>(model.max_covariates),
		    static_cast<unsigned long long>(out.features.size()),
		    static_cast<unsigned long long>(frame.X.cols)));
	}
	out.context_rows = SampleContext(frame, context_size, spec.seed, resample);
	return out;
}

//! Do-PFN: treatment is column 0 of a single X tensor, the output is logits over
//! a bar distribution, and the context length is fixed by the exported graph.
CfmResult RunDoPfn(ClientContext &context, const CausalFrame &frame, const CausalSpec &spec,
                   const ModelInfo &model, const string &dir) {
	CfmResult result;
	auto bars = LoadBorders(dir, model.manifest_file);
	if (!bars.loaded || bars.centres.size() != model.num_buckets) {
		throw BinderException("duckdo: could not read the bar-distribution borders from %s/%s. Re-run the export "
		                      "script to regenerate it",
		                      dir, model.manifest_file);
	}

	idx_t rung = 0;
	for (auto candidate : model.context_ladder) {
		if (candidate < frame.n && candidate > rung) {
			rung = candidate;
		}
	}
	if (rung == 0) {
		throw BinderException("duckdo: %llu rows is below the smallest context this model was exported for (%llu). "
		                      "Use a classical estimator on a table this small, or model := 'causalpfn', whose "
		                      "context length is not fixed",
		                      static_cast<unsigned long long>(frame.n),
		                      static_cast<unsigned long long>(model.context_ladder.front()));
	}
	result.ladder_rung = rung;

	auto prepared = PrepareCommon(frame, spec, model, rung, spec.ensemble > 1);
	result.warnings = prepared.warnings;
	for (auto j : prepared.features) {
		result.features_used.push_back(frame.features[j].name);
	}
	const auto &context_rows = prepared.context_rows;
	if (context_rows.size() != rung) {
		throw BinderException("duckdo: could not assemble a context of %llu rows from this table",
		                      static_cast<unsigned long long>(rung));
	}
	result.context_used = context_rows.size();

	// The model was trained on standardised outcomes, and its bar distribution
	// lives in that space. Standardise on the context, then scale back; the
	// centring cancels in the difference, but the scale does not.
	double y_mean = 0.0;
	for (auto r : context_rows) {
		y_mean += frame.y[r];
	}
	y_mean /= static_cast<double>(context_rows.size());
	double y_var = 0.0;
	for (auto r : context_rows) {
		const double d = frame.y[r] - y_mean;
		y_var += d * d;
	}
	const double y_sd = std::max(std::sqrt(y_var / static_cast<double>(context_rows.size() - 1)), 1e-12);

	const idx_t width = model.max_covariates + 1; // column 0 is the treatment
	std::vector<float> context_x(rung * width, 0.0f);
	std::vector<float> context_y(rung, 0.0f);
	for (idx_t i = 0; i < rung; i++) {
		const idx_t r = context_rows[i];
		context_x[i * width] = static_cast<float>(frame.t[r]);
		for (idx_t j = 0; j < prepared.features.size(); j++) {
			context_x[i * width + 1 + j] = static_cast<float>(frame.X.At(r, prepared.features[j]));
		}
		context_y[i] = static_cast<float>((frame.y[r] - y_mean) / y_sd);
	}

	auto &session = AcquireSession(GraphPath(dir, model, rung), NumericThreads());
	Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const std::array<int64_t, 3> context_shape {static_cast<int64_t>(rung), 1, static_cast<int64_t>(width)};
	const std::array<int64_t, 2> context_y_shape {static_cast<int64_t>(rung), 1};
	const char *input_names[] = {"context_x", "context_y", "query_x"};
	const char *output_names[] = {"logits"};

	const idx_t chunk = std::max<idx_t>(GetSettingIdx(context, "duckdo_query_chunk", 512), 1);
	result.cate.assign(frame.n, 0.0);

	for (idx_t start = 0; start < frame.n; start += chunk) {
		const idx_t count = std::min(chunk, frame.n - start);
		std::vector<float> query(count * width, 0.0f);
		for (idx_t i = 0; i < count; i++) {
			for (idx_t j = 0; j < prepared.features.size(); j++) {
				query[i * width + 1 + j] = static_cast<float>(frame.X.At(start + i, prepared.features[j]));
			}
		}
		const std::array<int64_t, 3> query_shape {static_cast<int64_t>(count), 1, static_cast<int64_t>(width)};

		std::vector<double> per_row[2];
		for (int arm = 0; arm < 2; arm++) {
			// do(T = arm): force column 0 for every query row. This is the
			// intervention - not a filter on rows where T happened to be arm.
			for (idx_t i = 0; i < count; i++) {
				query[i * width] = static_cast<float>(arm);
			}
			std::array<Ort::Value, 3> inputs {
			    Ort::Value::CreateTensor<float>(memory, context_x.data(), context_x.size(), context_shape.data(),
			                                    context_shape.size()),
			    Ort::Value::CreateTensor<float>(memory, context_y.data(), context_y.size(), context_y_shape.data(),
			                                    context_y_shape.size()),
			    Ort::Value::CreateTensor<float>(memory, query.data(), query.size(), query_shape.data(),
			                                    query_shape.size())};
			auto outputs =
			    session.Run(Ort::RunOptions {nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
			const float *logits = outputs[0].GetTensorData<float>();
			const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
			if (shape.size() != 3 || static_cast<idx_t>(shape[0]) != count ||
			    static_cast<idx_t>(shape[2]) != model.num_buckets) {
				throw BinderException("duckdo: model '%s' returned an unexpected output shape; the graph and the "
				                      "catalog entry disagree. Re-run the export script",
				                      model.id);
			}
			per_row[arm].assign(count, 0.0);
			for (idx_t i = 0; i < count; i++) {
				const float *row = logits + i * model.num_buckets;
				// Softmax over buckets, then the distribution's mean.
				float peak = row[0];
				for (idx_t b = 1; b < model.num_buckets; b++) {
					peak = std::max(peak, row[b]);
				}
				double total = 0.0, weighted = 0.0;
				for (idx_t b = 0; b < model.num_buckets; b++) {
					const double p = std::exp(static_cast<double>(row[b] - peak));
					total += p;
					weighted += p * bars.centres[b];
				}
				per_row[arm][i] = total > 0.0 ? weighted / total : 0.0;
			}
		}
		for (idx_t i = 0; i < count; i++) {
			// Back to the outcome's own scale. The centring cancels in the
			// difference; the scale does not.
			result.cate[start + i] = (per_row[1][i] - per_row[0][i]) * y_sd;
		}
	}
	return result;
}

//! CausalPFN: the treatment is its own tensor, the graph returns the conditional
//! expected potential outcome directly, and both lengths are dynamic - so there
//! is no ladder, no bar distribution and no outcome rescaling to undo. The model
//! standardises the outcome per arm internally.
CfmResult RunCausalPfn(ClientContext &context, const CausalFrame &frame, const CausalSpec &spec,
                       const ModelInfo &model, const string &dir) {
	CfmResult result;
	const idx_t context_size = std::min(frame.n, model.max_context);
	auto prepared = PrepareCommon(frame, spec, model, context_size, spec.ensemble > 1);
	result.warnings = prepared.warnings;
	for (auto j : prepared.features) {
		result.features_used.push_back(frame.features[j].name);
	}
	const auto &context_rows = prepared.context_rows;
	result.context_used = context_rows.size();
	result.ladder_rung = 0; // dynamic

	const idx_t width = model.max_covariates;
	const idx_t ctx = context_rows.size();
	std::vector<float> context_x(ctx * width, 0.0f);
	std::vector<float> context_t(ctx, 0.0f);
	std::vector<float> context_y(ctx, 0.0f);
	for (idx_t i = 0; i < ctx; i++) {
		const idx_t r = context_rows[i];
		for (idx_t j = 0; j < prepared.features.size(); j++) {
			context_x[i * width + j] = static_cast<float>(frame.X.At(r, prepared.features[j]));
		}
		context_t[i] = static_cast<float>(frame.t[r]);
		context_y[i] = static_cast<float>(frame.y[r]);
	}

	auto &session = AcquireSession(GraphPath(dir, model, 0), NumericThreads());
	Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const std::array<int64_t, 3> context_shape {1, static_cast<int64_t>(ctx), static_cast<int64_t>(width)};
	const std::array<int64_t, 2> context_1d {1, static_cast<int64_t>(ctx)};
	const char *input_names[] = {"X_context", "t_context", "y_context", "X_query", "t_query"};
	const char *output_names[] = {"mu"};

	const idx_t chunk = std::max<idx_t>(GetSettingIdx(context, "duckdo_query_chunk", 512), 1);
	result.cate.assign(frame.n, 0.0);

	for (idx_t start = 0; start < frame.n; start += chunk) {
		const idx_t count = std::min(chunk, frame.n - start);
		std::vector<float> query_x(count * width, 0.0f);
		for (idx_t i = 0; i < count; i++) {
			for (idx_t j = 0; j < prepared.features.size(); j++) {
				query_x[i * width + j] = static_cast<float>(frame.X.At(start + i, prepared.features[j]));
			}
		}
		const std::array<int64_t, 3> query_shape {1, static_cast<int64_t>(count), static_cast<int64_t>(width)};
		const std::array<int64_t, 2> query_1d {1, static_cast<int64_t>(count)};

		std::vector<double> per_row[2];
		for (int arm = 0; arm < 2; arm++) {
			// do(T = arm) for every query row.
			std::vector<float> query_t(count, static_cast<float>(arm));
			std::array<Ort::Value, 5> inputs {
			    Ort::Value::CreateTensor<float>(memory, context_x.data(), context_x.size(), context_shape.data(),
			                                    context_shape.size()),
			    Ort::Value::CreateTensor<float>(memory, context_t.data(), context_t.size(), context_1d.data(),
			                                    context_1d.size()),
			    Ort::Value::CreateTensor<float>(memory, context_y.data(), context_y.size(), context_1d.data(),
			                                    context_1d.size()),
			    Ort::Value::CreateTensor<float>(memory, query_x.data(), query_x.size(), query_shape.data(),
			                                    query_shape.size()),
			    Ort::Value::CreateTensor<float>(memory, query_t.data(), query_t.size(), query_1d.data(),
			                                    query_1d.size())};
			auto outputs =
			    session.Run(Ort::RunOptions {nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
			const float *mu = outputs[0].GetTensorData<float>();
			const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
			idx_t produced = 1;
			for (auto dim : shape) {
				produced *= static_cast<idx_t>(dim);
			}
			if (produced != count) {
				throw BinderException("duckdo: model '%s' returned %llu values for %llu query rows; the graph and "
				                      "the catalog entry disagree. Re-run the export script",
				                      model.id, static_cast<unsigned long long>(produced),
				                      static_cast<unsigned long long>(count));
			}
			per_row[arm].assign(count, 0.0);
			for (idx_t i = 0; i < count; i++) {
				per_row[arm][i] = static_cast<double>(mu[i]);
			}
		}
		for (idx_t i = 0; i < count; i++) {
			result.cate[start + i] = per_row[1][i] - per_row[0][i];
		}
	}
	return result;
}

} // namespace

CfmResult CfmEstimateCate(ClientContext &context, const CausalFrame &frame, const CausalSpec &spec,
                          const ModelInfo &model) {
	const string dir = ModelDir(context);
	string missing;
	if (!ModelArtifactsPresent(context, model, missing)) {
		const char *script = model.kind == ModelKind::CAUSALPFN ? "export_causalpfn.py --out %s"
		                                                        : "export_dopfn.py --repo <Do-PFN checkout> --out %s";
		throw BinderException("duckdo: model '%s' is not available: '%s' is missing from %s. Export it with "
		                      "`python scripts/export/%s`",
		                      model.id, missing, dir, StringUtil::Format(script, dir));
	}

	const idx_t draws = std::min<idx_t>(std::max<idx_t>(spec.ensemble, 1), 25);
	auto run_once = [&](const CausalSpec &draw_spec) {
		return model.kind == ModelKind::CAUSALPFN ? RunCausalPfn(context, frame, draw_spec, model, dir)
		                                          : RunDoPfn(context, frame, draw_spec, model, dir);
	};

	auto result = run_once(spec);
	result.draws = 1;

	if (draws > 1) {
		// Which rows land in the context is a real source of uncertainty, and it
		// is one a single forward pass cannot see. Re-drawing the context and
		// looking at the spread measures it directly. It is not full model
		// uncertainty - the weights are fixed - and the variance method says so.
		vector<vector<double>> all;
		all.reserve(draws);
		all.push_back(result.cate);
		vector<double> ate_draws;
		double sum = 0.0;
		for (auto v : result.cate) {
			sum += v;
		}
		ate_draws.push_back(sum / static_cast<double>(frame.n));

		for (idx_t d = 1; d < draws; d++) {
			CausalSpec draw_spec = spec;
			draw_spec.seed = spec.seed + static_cast<int64_t>(d) * 7919;
			auto extra = run_once(draw_spec);
			double draw_sum = 0.0;
			for (auto v : extra.cate) {
				draw_sum += v;
			}
			ate_draws.push_back(draw_sum / static_cast<double>(frame.n));
			all.push_back(std::move(extra.cate));
		}

		result.draws = draws;
		result.cate.assign(frame.n, 0.0);
		result.cate_se.assign(frame.n, 0.0);
		for (idx_t i = 0; i < frame.n; i++) {
			double mean = 0.0;
			for (auto &draw : all) {
				mean += draw[i];
			}
			mean /= static_cast<double>(draws);
			double variance = 0.0;
			for (auto &draw : all) {
				const double diff = draw[i] - mean;
				variance += diff * diff;
			}
			variance /= static_cast<double>(draws - 1);
			result.cate[i] = mean;
			result.cate_se[i] = std::sqrt(variance);
		}

		const double ate_mean = Mean(ate_draws);
		double ate_var = 0.0;
		for (auto v : ate_draws) {
			const double diff = v - ate_mean;
			ate_var += diff * diff;
		}
		result.ate_between_draw_se = std::sqrt(ate_var / static_cast<double>(draws - 1));
		if (!(result.ate_between_draw_se > 1e-12)) {
			// Every draw came back identical, so the ensemble learned nothing and
			// the interval it would imply is fiction.
			result.draws = 1;
			result.cate_se.clear();
			result.ate_between_draw_se = 0.0;
			result.warnings.push_back("the context draws were identical, so the ensemble measured no uncertainty "
			                          "and no interval is reported");
		} else {
			result.warnings.push_back(StringUtil::Format(
			    "%llu bootstrapped context draws; the interval covers context selection and sampling, not the "
			    "model's own parameter uncertainty",
			    static_cast<unsigned long long>(draws)));
		}
	}

	if (model.attribution_required) {
		result.warnings.push_back(model.id + " is " + model.license + "; attribution is required downstream");
	}
	return result;
}

#endif // DUCKDO_WITH_ONNX

// --- SQL surface ------------------------------------------------------------

namespace {

struct ModelGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<ModelGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<ModelGlobalState>();
	idx_t count = 0;
	while (state.offset < bind.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &row = bind.rows[state.offset];
		for (idx_t c = 0; c < row.size(); c++) {
			output.SetValue(c, count, row[c]);
		}
		state.offset++;
		count++;
	}
	output.SetCardinality(count);
}

unique_ptr<FunctionData> BindListModels(ClientContext &context, TableFunctionBindInput &,
                                        vector<LogicalType> &return_types, vector<string> &names) {
	names = {"model", "setting", "license", "commercial", "attribution_required",
	         "max_covariates", "context_ladder", "available", "detail"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::BOOLEAN, LogicalType::BOOLEAN, LogicalType::BIGINT,
	                LogicalType::LIST(LogicalType::BIGINT), LogicalType::BOOLEAN, LogicalType::VARCHAR};
	auto bind = make_uniq<ResultBindData>();
	const string dir = ModelDir(context);
	for (auto &model : ModelCatalog()) {
		vector<Value> ladder;
		for (auto rung : model.context_ladder) {
			ladder.push_back(Value::BIGINT(static_cast<int64_t>(rung)));
		}
		string missing;
		bool available = false;
		string detail;
		if (!OnnxAvailable()) {
			detail = "this build has no ONNX Runtime; rebuild with -DDUCKDO_ONNXRUNTIME_ROOT";
		} else if (!ModelArtifactsPresent(context, model, missing)) {
			detail = "'" + missing + "' missing from " + dir;
		} else {
			available = true;
			detail = "ready in " + dir;
		}
		if (model.attribution_required) {
			detail += "; " + model.license + " requires attribution downstream";
		}
		bind->rows.push_back({Value(model.id), Value(model.setting), Value(model.license),
		                      Value::BOOLEAN(model.commercial), Value::BOOLEAN(model.attribution_required),
		                      Value::BIGINT(static_cast<int64_t>(model.max_covariates)),
		                      Value::LIST(LogicalType::BIGINT, std::move(ladder)), Value::BOOLEAN(available),
		                      Value(detail)});
	}
	return std::move(bind);
}

unique_ptr<FunctionData> BindDevices(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                     vector<string> &names) {
	names = {"device", "available"};
	return_types = {LogicalType::VARCHAR, LogicalType::BOOLEAN};
	auto bind = make_uniq<ResultBindData>();
	const auto devices = AvailableDevices();
	for (auto &device : devices) {
		bind->rows.push_back({Value(device), Value::BOOLEAN(true)});
	}
	if (devices.empty()) {
		bind->rows.push_back({Value("cpu"), Value::BOOLEAN(false)});
	}
	for (auto gpu : {"cuda", "rocm", "mlx"}) {
		bind->rows.push_back({Value(gpu), Value::BOOLEAN(false)});
	}
	return std::move(bind);
}

} // namespace

void RegisterModelFunctions(ExtensionLoader &loader) {
	TableFunction models("", vector<LogicalType>(), EmitRows, BindListModels, InitGlobal);
	RegisterUnderBothNames(loader, models, "list_models");
	TableFunction models_alias("", vector<LogicalType>(), EmitRows, BindListModels, InitGlobal);
	RegisterUnderBothNames(loader, models_alias, "models");

	TableFunction devices("", vector<LogicalType>(), EmitRows, BindDevices, InitGlobal);
	RegisterUnderBothNames(loader, devices, "devices");
}

} // namespace duckdo
} // namespace duckdb
