#pragma once

#include "ann_data.h"
#include <string>
#include <memory>

// Forward-declare HighFive::File for private method signatures
namespace HighFive { class File; }

class H5adReader {
public:
    // Read an h5ad file and return an AnnDataContainer
    // Only reads obsm, obs, obsp (skips X for memory efficiency)
    static std::shared_ptr<AnnDataContainer> read(const std::string& path);

    // Read from sparse matrix text/binary files (alternative to h5ad)
    // Expected file structure:
    //   folder/umap.txt      (n_obs x 2, tab-separated)
    //   folder/obs.csv       (header + n_obs rows)
    //   folder/connectivities.mtx  (sparse matrix in MatrixMarket format, optional)
    static std::shared_ptr<AnnDataContainer> readFromMatrixFiles(const std::string& folder_path);

private:
    static void readObsm(HighFive::File& file, AnnDataContainer& adata);
    static void readObs(HighFive::File& file, AnnDataContainer& adata);
    static void readObsp(HighFive::File& file, AnnDataContainer& adata);
};
