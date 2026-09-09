//===----------------------------------------------------------------------===//
//                         DuckDo - causal inference in DuckDB
//
// duckdo/functions.hpp
//
// SQL surface registration. Every function is registered twice: under its
// short `do_*` name and its unambiguous `duckdo_*` full name.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace duckdo {

//! Rows precomputed at bind time, emitted verbatim during execution. Every
//! DuckDo table function is small-output or bounded by duckdo_max_rows, so
//! materialising in bind keeps the execution path trivial.
struct ResultBindData : public TableFunctionData {
	vector<vector<Value>> rows;

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;
};

//! Attach the named parameters shared by every estimation function.
void AddCommonNamedParameters(TableFunction &fn);

void RegisterEstimationFunctions(ExtensionLoader &loader);
void RegisterInterventionFunctions(ExtensionLoader &loader);
void RegisterContinuousFunctions(ExtensionLoader &loader);
void RegisterPanelFunctions(ExtensionLoader &loader);
void RegisterDiagnosticFunctions(ExtensionLoader &loader);
void RegisterGraphFunctions(ExtensionLoader &loader);
//! Point the graph store at the database it should persist into.
void SetGraphDatabase(DatabaseInstance &instance);

//! Register a table function under both `do_<name>` and `duckdo_<name>`.
void RegisterUnderBothNames(ExtensionLoader &loader, TableFunction fn, const string &bare_name);

} // namespace duckdo
} // namespace duckdb
