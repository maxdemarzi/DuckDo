#define DUCKDB_EXTENSION_MAIN

#include "duckdo_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdo/functions.hpp"
#include "duckdo/runtime.hpp"

namespace duckdb {

static void RegisterSettings(DatabaseInstance &instance) {
	auto &config = DBConfig::GetConfig(instance);
	config.AddExtensionOption("duckdo_default_estimator",
	                          "Estimator used when a function is called without estimator := "
	                          "(naive, regression, ipw, aipw, dml, s_learner, t_learner, x_learner, dr_learner)",
	                          LogicalType::VARCHAR, Value("aipw"));
	config.AddExtensionOption("duckdo_max_rows",
	                          "Maximum number of rows an estimation frame may contain before DuckDo refuses to "
	                          "run. The frame is held in memory as doubles, so 1M rows by 50 covariates is "
	                          "roughly 400 MB",
	                          LogicalType::BIGINT, Value::BIGINT(1000000));
	config.AddExtensionOption("duckdo_max_features",
	                          "Maximum number of encoded features an estimation frame may contain",
	                          LogicalType::BIGINT, Value::BIGINT(500));
	config.AddExtensionOption("duckdo_max_categorical_levels",
	                          "Categorical covariates with more distinct levels than this are dropped with a warning",
	                          LogicalType::BIGINT, Value::BIGINT(32));
	config.AddExtensionOption("duckdo_max_groups",
	                          "Maximum number of groups do_ate_by will estimate before refusing to run",
	                          LogicalType::BIGINT, Value::BIGINT(1000));
	config.AddExtensionOption("duckdo_model_dir",
	                          "Directory holding exported causal foundation model graphs and weights "
	                          "(default ~/.cache/duckdo)",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("duckdo_threads",
	                          "Threads for model inference and for the dense accumulations inside every "
	                          "estimator; 0 means one per hardware thread",
	                          LogicalType::BIGINT, Value::BIGINT(0));
	config.AddExtensionOption("duckdo_ensemble_draws",
	                          "Independent context draws a foundation model takes. One is a single pass with no "
	                          "interval; more than one funds one from the spread across draws",
	                          LogicalType::BIGINT, Value::BIGINT(1));
	config.AddExtensionOption("duckdo_query_chunk",
	                          "Rows scored per model forward pass", LogicalType::BIGINT, Value::BIGINT(512));
	config.AddExtensionOption("duckdo_seed", "Seed for fold assignment, bootstrap and every other random draw",
	                          LogicalType::BIGINT, Value::BIGINT(42));
	config.AddExtensionOption("duckdo_bootstrap_reps",
	                          "Bootstrap replicates used by estimators without a closed-form influence function",
	                          LogicalType::BIGINT, Value::BIGINT(200));
}

static void LoadInternal(ExtensionLoader &loader) {
	loader.SetDescription("Causal inference in DuckDB: treatment effects, diagnostics, graph identification and "
	                      "interventional queries in SQL");
	RegisterSettings(loader.GetDatabaseInstance());
	duckdo::RegisterEstimationFunctions(loader);
	duckdo::RegisterInterventionFunctions(loader);
	duckdo::RegisterContinuousFunctions(loader);
	duckdo::RegisterPanelFunctions(loader);
	duckdo::RegisterDiagnosticFunctions(loader);
	duckdo::RegisterGraphFunctions(loader);
	duckdo::RegisterModelFunctions(loader);
}

void DuckdoExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string DuckdoExtension::Name() {
	return "duckdo";
}

std::string DuckdoExtension::Version() const {
#ifdef EXT_VERSION_DUCKDO
	return EXT_VERSION_DUCKDO;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckdo, loader) {
	duckdb::LoadInternal(loader);
}
}
