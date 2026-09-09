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

//! One entry in the built-in model catalog.
struct ModelInfo {
	string id;
	string setting;
	string license;
	string source;
	//! CC BY 4.0 is commercially usable but obliges attribution downstream.
	bool commercial = true;
	bool attribution_required = false;
	//! Total input columns the graph accepts, treatment included.
	idx_t max_features = 0;
	idx_t num_buckets = 0;
	//! Fixed context sizes the exported graphs were traced at, ascending.
	vector<idx_t> context_ladder;
	string weights_file;
	string graph_pattern;
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
