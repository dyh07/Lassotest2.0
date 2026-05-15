#pragma once

#include "data_store.h"
#include <string>
#include <memory>
#include <thread>

// Forward declaration for Crow
// We use Crow as a header-only HTTP framework

class HttpServer {
public:
    HttpServer(DatasetStore& dataset_store, JobStore& job_store, int port = 15114);
    ~HttpServer();

    void start();  // Start in a separate thread
    void stop();

private:
    DatasetStore& m_dataset_store;
    JobStore& m_job_store;
    int m_port;
    std::unique_ptr<std::thread> m_thread;
    bool m_running = false;

    // Helper functions for building JSON responses
    nlohmann::json buildDatasetSummary(const QString& dataset_id, std::shared_ptr<AnnDataContainer> adata);
    nlohmann::json buildPlotPayload(std::shared_ptr<AnnDataContainer> adata,
                                      const std::string& embedding_key,
                                      const std::string& color_by);

    std::string resolveEmbeddingKey(std::shared_ptr<AnnDataContainer> adata, const std::string& requested);
    std::string resolveColorColumn(std::shared_ptr<AnnDataContainer> adata, const std::string& requested);
};
