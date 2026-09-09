//===----------------------------------------------------------------------===//
//                         DuckDo - causal inference in DuckDB
//
// duckdo/runtime.hpp
//
// Phase 5-6: the causal foundation model path.
//
// Built only when DUCKDO_WITH_ONNX is defined, so the default build stays
// dependency-free and the community extension keeps its "no downloads" first
// impression. Without it every entry point here still exists and reports,
// specifically, that the build lacks ONNX support - never a silent fallback.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"
#include "duckdo/frame.hpp"

namespace duckdb {
namespace duckdo {

//! How a model's graph is laid out. The two families differ enough that they
//! need separate marshalling, but nothing above this layer has to care.
enum class ModelKind : uint8_t {
	//! Treatment is column 0 of a single X tensor; output is logits over a bar
	//! distribution; the context length is fixed per exported graph.
	DOPFN,
	//! Treatment is its own tensor; output is the conditional expected potential
	//! outcome directly; both context and query lengths are dynamic.
	CAUSALPFN
};

//! One entry in the built-in model catalog.
struct ModelInfo {
	ModelKind kind = ModelKind::DOPFN;
	string id;
	string setting;
	string license;
	string source;
	//! CC BY 4.0 is commercially usable but obliges attribution downstream.
	bool commercial = true;
	bool attribution_required = false;
	//! Covariates the graph accepts, treatment excluded.
	idx_t max_covariates = 0;
	//! Bar-distribution buckets. Zero when the model returns a mean directly.
	idx_t num_buckets = 0;
	//! Largest context the model accepts. Zero means "see context_ladder".
	idx_t max_context = 0;
	//! Fixed context sizes the exported graphs were traced at, ascending.
	vector<idx_t> context_ladder;
	string weights_file;
	//! printf pattern for a ladder graph, or a plain filename when there is one.
	string graph_pattern;
	string manifest_file;
};

//! The catalog. Static: adding a model means shipping a new graph.
const vector<ModelInfo> &ModelCatalog();
//! nullptr when the id is unknown.
const ModelInfo *FindModel(const string &id);
//! Comma-separated ids, for error messages.
string KnownModels();

//! Where graphs and weights live. duckdo_model_dir, else ~/.cache/duckdo.
string ModelDir(ClientContext &context);

//! True when every artifact the model needs is present on disk.
bool ModelArtifactsPresent(ClientContext &context, const ModelInfo &model, string &missing);

//! Whether this build can run models at all.
bool OnnxAvailable();
//! Execution providers this build can offer.
vector<string> AvailableDevices();

//! Per-row conditional effects from a causal foundation model.
struct CfmResult {
	vector<double> cate;
	//! Standard error of each row's effect across the context ensemble. Empty
	//! when only one draw was taken, because one draw funds no interval.
	vector<double> cate_se;
	//! Context draws actually taken.
	idx_t draws = 1;
	//! Standard error of the population effect across draws, which is the part
	//! a single draw cannot see at all.
	double ate_between_draw_se = 0.0;
	//! Context rows actually fed to the model.
	idx_t context_used = 0;
	//! Ladder rung selected.
	idx_t ladder_rung = 0;
	//! Covariates kept after fitting the model's feature budget.
	vector<string> features_used;
	vector<string> warnings;
};

//! Run the model over the frame and return one effect per row.
//! Throws a BinderException naming the remediation on any failure.
CfmResult CfmEstimateCate(ClientContext &context, const CausalFrame &frame, const CausalSpec &spec,
                          const ModelInfo &model);

void RegisterModelFunctions(ExtensionLoader &loader);

} // namespace duckdo
} // namespace duckdb
