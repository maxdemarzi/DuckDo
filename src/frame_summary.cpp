//===----------------------------------------------------------------------===//
// do_frame_summary: what the encoder actually did.
//
// The roadmap names this as the mitigation for a specific risk - "encoder
// choices can silently bias estimates" - and every estimator in DuckDo runs on
// the frame this reports. Without it, questions like "which columns did it
// adjust for?", "was that categorical one-hot encoded or dropped?" and "how many
// rows had a missing value filled in?" are answerable only by reading the
// source.
//
// One row per encoded feature, plus a header row for the frame itself, so a
// single call answers both "what is in here" and "what became of column X".
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"

namespace duckdb {
namespace duckdo {

namespace {

struct SummaryGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<SummaryGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<SummaryGlobalState>();
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

const char *KindName(FeatureKind kind) {
	switch (kind) {
	case FeatureKind::ONE_HOT:
		return "one_hot";
	case FeatureKind::MISSING_INDICATOR:
		return "missing_indicator";
	default:
		return "numeric";
	}
}

unique_ptr<FunctionData> BindFrameSummary(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	auto frame = BuildFrame(context, spec);

	names = {"feature", "source", "kind", "level", "center", "scale", "n", "n_features", "n_treated",
	         "n_rows_with_missing", "treated_label", "control_label", "warnings"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::BIGINT,  LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::LIST(LogicalType::VARCHAR)};

	vector<Value> warning_values;
	for (auto &w : frame.warnings) {
		warning_values.push_back(Value(w));
	}
	auto warning_list = Value::LIST(LogicalType::VARCHAR, std::move(warning_values));

	// The frame-level counts repeat on every row rather than living in a separate
	// result, so a single call answers both kinds of question and an empty
	// feature set still reports the frame.
	auto row_for = [&](const Value &feature, const Value &source, const Value &kind, const Value &level,
	                   const Value &center, const Value &scale) {
		return vector<Value>{feature,
		                     source,
		                     kind,
		                     level,
		                     center,
		                     scale,
		                     Value::BIGINT(static_cast<int64_t>(frame.n)),
		                     Value::BIGINT(static_cast<int64_t>(frame.features.size())),
		                     Value::BIGINT(static_cast<int64_t>(frame.n_treated)),
		                     Value::BIGINT(static_cast<int64_t>(frame.n_rows_with_missing)),
		                     Value(frame.treated_label),
		                     Value(frame.control_label),
		                     warning_list};
	};

	auto bind = make_uniq<ResultBindData>();
	if (frame.features.empty()) {
		// No covariates is a legitimate answer - `covariates := []` asks for it -
		// and the frame still has a size, two arms and any warnings.
		bind->rows.push_back(row_for(Value(LogicalType::VARCHAR), Value(LogicalType::VARCHAR),
		                             Value("no covariates"), Value(LogicalType::VARCHAR),
		                             Value(LogicalType::DOUBLE), Value(LogicalType::DOUBLE)));
		return std::move(bind);
	}
	for (auto &feature : frame.features) {
		bind->rows.push_back(row_for(Value(feature.name), Value(feature.source), Value(KindName(feature.kind)),
		                             feature.level.empty() ? Value(LogicalType::VARCHAR) : Value(feature.level),
		                             Value::DOUBLE(feature.center), Value::DOUBLE(feature.scale)));
	}
	return std::move(bind);
}

} // namespace

void RegisterFrameSummaryFunction(ExtensionLoader &loader) {
	TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, BindFrameSummary, InitGlobal);
	AddCommonNamedParameters(fn);
	RegisterUnderBothNames(loader, fn, "frame_summary");
}

} // namespace duckdo
} // namespace duckdb
