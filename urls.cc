#include <iostream>
#include <thread>

// xgboost
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/learner.h>

#include "data/adapter.h"

// csvparser
#include "single_include/csv.hpp"

using namespace xgboost;

template <typename T>
std::vector<T> read_csv(const std::string path) {
  std::vector<T> values;

  csv::CSVReader reader(path);

  for (const csv::CSVRow& row : reader) {
    values.push_back(row[0].get<T>());
  }

  return values;
}

/// Use out own parsing because the xgboost devs recommend not to use their own
/// DMatrix::Load function: "Currently, the DMLC data parser cannot parse CSV
/// files with headers. Use Pandas (see below) to read CSV files with headers."
/// see https://xgboost.readthedocs.io/en/release_1.2.0/python/python_intro.html
data::DenseAdapter csv_to_adapter(const std::string path,
                                  std::vector<float>& values,
                                  std::vector<std::string>& col_names) {
  csv::CSVReader reader(path);

  for (const csv::CSVRow& row : reader) {
    for (csv::CSVField& field : row) {
      values.push_back(field.get<float>());
    }
  }

  col_names = reader.get_col_names();
  const size_t ncol = col_names.size();
  const size_t nrow = values.size() / ncol;

  return data::DenseAdapter(values.data(), nrow, ncol);
}

data::DenseAdapter csv_to_adapter(const std::string path,
                                  std::vector<float>& values) {
  std::vector<std::string> col_names;
  return csv_to_adapter(path, values, col_names);
}

/// Json outout
void print_learner_json(const std::shared_ptr<Learner>& learner) {
  Json config{Object()};
  learner->Configure();
  learner->SaveConfig(&config);
  std::string& config_str = learner->GetThreadLocal().ret_str;
  Json::Dump(config, &config_str);
  std::cout << config_str << std::endl;
}

int main() {
  const std::string x_train_path = "../data/malicious_phish_X_train.csv";
  const std::string x_test_path = "../data/malicious_phish_X_test.csv";
  const std::string y_train_path = "../data/malicious_phish_y_train.csv";
  const std::string y_test_path = "../data/malicious_phish_y_test.csv";
  constexpr size_t n_estimators = 100;  // The number of rounds for boosting

  std::vector<std::string> col_names;
  std::vector<float> train_values;
  data::DenseAdapter train_adapter =
      csv_to_adapter(x_train_path, train_values, col_names);
  const auto train = std::shared_ptr<DMatrix>(DMatrix::Create(
      &train_adapter, /*missing=*/std::numeric_limits<float>::quiet_NaN(),
      /*nthread=*/std::thread::hardware_concurrency()));
  const std::vector<float> train_labels = read_csv<float>(y_train_path);
  train->SetInfo("label", train_labels.data(), xgboost::DataType::kFloat32,
                 train_labels.size());

  std::shared_ptr<Learner> learner{Learner::Create({train})};
  learner->SetParam("objective", "binary:logistic");
  learner->SetFeatureNames(col_names);
  learner->Configure();

  std::vector<float> test_values;
  data::DenseAdapter test_adapter = csv_to_adapter(x_test_path, test_values);
  const auto test = std::shared_ptr<DMatrix>(DMatrix::Create(
      &test_adapter, /*missing=*/std::numeric_limits<float>::quiet_NaN(),
      /*nthread=*/std::thread::hardware_concurrency()));
  const std::vector<float> test_labels = read_csv<float>(y_test_path);
  test->SetInfo("label", test_labels.data(), xgboost::DataType::kFloat32,
                test_labels.size());

#ifndef NDEBUG
  const auto num_feature = learner->GetNumFeature();
  std::cout << "num_feature: " << num_feature << std::endl;

  std::vector<std::string>& str_vecs = learner->GetThreadLocal().ret_vec_str;
  learner->GetFeatureNames(&str_vecs);
  std::cout << "feature names: ";
  for (const auto& str_vec : str_vecs) {
    std::cout << str_vec << ' ';
  }
  std::cout << std::endl;

  learner->GetFeatureTypes(&str_vecs);
  std::cout << "feature types: ";
  for (const auto& str_vec : str_vecs) {
    std::cout << str_vec << ' ';
  }
  std::cout << std::endl;

  str_vecs = learner->GetAttrNames();
  std::cout << "attr names: ";
  for (const auto& str_vec : str_vecs) {
    std::cout << str_vec << ' ';
  }
  std::cout << std::endl;

  print_learner_json(learner);
#endif

  auto train_start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < n_estimators; iter++) {
    learner->UpdateOneIter(iter, train);

#ifndef NDEBUG
    if (iter % 10 == 0 || iter == n_estimators - 1) {
      const std::string res = learner->EvalOneIter(iter, {test}, {"test"});
      std::cout << res << std::endl;
    }
#endif
  }
  auto train_end = std::chrono::high_resolution_clock::now();
  auto train_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(train_end -
                                                                train_start);
  std::cout << "Training " << train->Info().num_row_ << " samples for "
            << n_estimators << " iterations took " << train_seconds.count()
            << " seconds." << std::endl;

  auto& entry = learner->GetThreadLocal().prediction_entry;
  auto iteration_end = GetIterationFromTreeLimit(0, learner.get());

  auto test_start = std::chrono::high_resolution_clock::now();
  learner->Predict(
      /*data=*/test, /*output_margin =*/false, /*out_preds=*/&entry.predictions,
      /*layer_begin=*/0, /*layer_end=*/iteration_end);
  auto test_end = std::chrono::high_resolution_clock::now();
  auto test_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
      test_end - test_start);
  std::cout << "Tesing " << test->Info().num_row_ << " samples took "
            << test_seconds.count() << " seconds." << std::endl;
#ifndef NDEBUG
  std::cout << "#predictions: " << entry.predictions.Size() << std::endl;
  std::cout << "#labels: " << test_labels.size() << std::endl;

  for (size_t i = 0; i < 100; i++)
    std::cout << "label[" << i << "]=" << test_labels[i] << " prediction[" << i
              << "]=" << entry.predictions.ConstHostVector()[i] << std::endl;
#endif

  size_t tp = 0, fp = 0, tn = 0, fn = 0;
  for (size_t i = 0; i < entry.predictions.Size(); i++) {
    const bool label = std::round(test_labels[i]);
    const bool pred = std::round(entry.predictions.ConstHostVector()[i]);

    tp += (label && pred);
    fp += (label && !pred);
    tn += (!label && !pred);
    fn += (!label && pred);
  }

  std::cout << std::endl;
  std::cout << "         "
            << "Positive "
            << "Negative " << std::endl;
  std::cout << "Positive " << std::setw(9) << tp << std::setw(9) << fp
            << std::endl;
  std::cout << "Negative " << std::setw(9) << fn << std::setw(9) << tn
            << std::endl;
  std::cout << std::endl;
  std::cout << "Accuracy: " << static_cast<double>(tp + tn) / test_labels.size()
            << std::endl;

  return 0;
}