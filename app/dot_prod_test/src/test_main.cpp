// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2015.02.12

#include <vector>
#include <ml/include/ml.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

DEFINE_int32(dim, 1e6, "Feature dimension");
DEFINE_int32(num_prod, 1e3, "# of dot products for each test.");
DEFINE_int32(nnz, 100, "# of nonzeros in sparse vec.");

void RunDenseFeature(const std::vector<std::vector<float>>& vecs) {
  std::vector<petuum::ml::DenseFeature<float>> dense_features(FLAGS_num_prod);
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    dense_features[i] = petuum::ml::DenseFeature<float>(vecs[i]);
  }

  petuum::HighResolutionTimer timer;
  double total = 0.;
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    total += petuum::ml::DenseDenseFeatureDotProduct(dense_features[0],
        dense_features[i]);
  }
  LOG(INFO) << "DenseFeature finishes in " << timer.elapsed()
    << "s. Result: " << total;
}

void RunEigenDenseFeature(const std::vector<std::vector<float>>& vecs) {
  double total = 0.;
  Eigen::Map<const Eigen::VectorXf> e1(vecs[0].data(), vecs[0].size());
  petuum::HighResolutionTimer timer;
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    Eigen::Map<const Eigen::VectorXf> ei(vecs[i].data(), vecs[i].size());
    total += e1.dot(ei);
  }
  LOG(INFO) << "EigenDenseFeature finishes in " << timer.elapsed()
    << "s. Result: " << total;
}

void RunSparseFeature(const std::vector<float>& dense_w,
    const std::vector<std::vector<int>>& sparse_idx,
    const std::vector<std::vector<float>>& sparse_val) {
  petuum::ml::DenseFeature<float> w(dense_w);
  std::vector<petuum::ml::SparseFeature<float>> sparse_features(FLAGS_num_prod);
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    sparse_features[i] = petuum::ml::SparseFeature<float>(sparse_idx[i],
        sparse_val[i], FLAGS_dim);
  }

  petuum::HighResolutionTimer timer;
  double total = 0.;
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    total += petuum::ml::DenseSparseFeatureDotProduct(w,
        sparse_features[i]);
  }
  LOG(INFO) << "SparseFeature finishes in " << timer.elapsed()
    << "s. Result: " << total;
}

void RunEigenSparseFeature(const std::vector<float>& dense_w,
    const std::vector<std::vector<int>>& sparse_idx,
    const std::vector<std::vector<float>>& sparse_val) {
  Eigen::Map<const Eigen::VectorXf> e1(dense_w.data(), dense_w.size());
  std::vector<Eigen::SparseVector<float>> sparse_features(FLAGS_num_prod);
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    /*
    std::vector<Eigen::Triplet<float>> triplets(FLAGS_nnz);
    for (int j = 0; j < FLAGS_nnz; ++j) {
      triplets[j] = Eigen::Triplet<float>(sparse_idx[i][j], 0, sparse_val[i][j]);
    }
    sparse_features[i].setFromTriplets(triplets.begin(), triplets.end());
    */
    for (int j = 0; j < FLAGS_nnz; ++j) {
      sparse_features[i].coeffRef(sparse_idx[i][j]) = sparse_val[i][j];
    }
  }
  petuum::HighResolutionTimer timer;
  double total = 0.;
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    Eigen::VectorXf v = e1 * sparse_features[i];
    total += v.sum();
  }
  LOG(INFO) << "EigenSparseFeature finishes in " << timer.elapsed()
    << "s. Result: " << total;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  petuum::HighResolutionTimer init_timer;
  // Initialize dense vectors
  std::vector<std::vector<float>> vecs(FLAGS_num_prod);
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    vecs[i] = std::vector<float>(FLAGS_dim);
    for (int j = 0; j < FLAGS_dim; ++j) {
      vecs[i][j] = i * 1.1 + j / 100.;
    }
  }
  LOG(INFO) << "Init Dense in " << init_timer.elapsed() << "s";

  RunDenseFeature(vecs);
  RunEigenDenseFeature(vecs);

  // Initialize sparse vectors
  petuum::HighResolutionTimer init_sparse_timer;
  std::vector<std::vector<int>> sparse_idx(FLAGS_num_prod);
  std::vector<std::vector<float>> sparse_val(FLAGS_num_prod);
  for (int i = 0; i < FLAGS_num_prod; ++i) {
    sparse_idx[i] = std::vector<int>(FLAGS_nnz);
    sparse_val[i] = std::vector<float>(FLAGS_nnz);
    for (int j = 0; j < FLAGS_nnz; ++j) {
      sparse_idx[i][j] = 11 + j * 5;
      sparse_val[i][j] = i * 1.1 + j / 100.;
    }
  }
  LOG(INFO) << "Init Sparse in " << init_sparse_timer.elapsed() << "s";

  RunSparseFeature(vecs[0], sparse_idx, sparse_val);
  RunEigenSparseFeature(vecs[0], sparse_idx, sparse_val);

  LOG(INFO) << "Test finished.";
  return 0;
}
