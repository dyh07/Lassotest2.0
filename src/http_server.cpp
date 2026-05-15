#include "http_server.h"
#include "h5ad_reader.h"
#include "lasso_view.h"
#include "downsample.h"
#include "json_utils.h"

// Qt defines 'signals' as a macro (to 'public'), which conflicts with
// Crow's signals() method. Undefine it before including Crow.
#pragma push_macro("signals")
#undef signals
#include <crow.h>
#pragma pop_macro("signals")

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QUuid>
#include <QDateTime>
#include <QThread>
#include <QtConcurrent>
#include <set>

// C++17-compatible ends_with helper
static bool str_ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

using json = nlohmann::json;

// Helper to create a JSON response with proper content-type
static crow::response json_response(int code, const std::string& body) {
    crow::response resp(code, body);
    resp.set_header("Content-Type", "application/json");
    return resp;
}

static crow::response json_response(const std::string& body) {
    crow::response resp(body);
    resp.set_header("Content-Type", "application/json");
    return resp;
}

HttpServer::HttpServer(DatasetStore& dataset_store, JobStore& job_store, int port)
    : m_dataset_store(dataset_store), m_job_store(job_store), m_port(port) {}

HttpServer::~HttpServer() {
    stop();
}

std::string HttpServer::resolveEmbeddingKey(std::shared_ptr<AnnDataContainer> adata, const std::string& requested) {
    if (!requested.empty()) {
        if (adata->obsm.count(requested)) return requested;
    }
    auto def = adata->defaultEmbeddingKey();
    return def.empty() ? "" : def;
}

std::string HttpServer::resolveColorColumn(std::shared_ptr<AnnDataContainer> adata, const std::string& requested) {
    if (!requested.empty()) {
        if (adata->obs.count(requested)) return requested;
    }
    auto def = adata->defaultColorColumn();
    return def.empty() ? "" : def;
}

json HttpServer::buildDatasetSummary(const QString& dataset_id, std::shared_ptr<AnnDataContainer> adata) {
    auto meta = m_dataset_store.getMeta(dataset_id);
    json j;
    j["dataset_id"] = dataset_id.toStdString();
    j["dataset_name"] = meta.dataset_name.toStdString();
    j["n_obs"] = adata->n_obs;
    j["n_vars"] = adata->n_vars;

    auto obs_cols = adata->getObsColumns();
    j["obs_columns"] = json::array();
    for (const auto& c : obs_cols) j["obs_columns"].push_back(c);

    auto embeddings = adata->getAvailableEmbeddings();
    j["available_embeddings"] = json::array();
    for (const auto& e : embeddings) j["available_embeddings"].push_back(e);

    j["default_embedding"] = adata->defaultEmbeddingKey();
    j["default_color_by"] = adata->defaultColorColumn();
    j["needs_umap_choice"] = adata->defaultEmbeddingKey().empty();
    j["has_other_embeddings"] = !embeddings.empty();
    j["is_derived"] = meta.is_derived;
    j["analysis_type"] = meta.analysis_type.toStdString();
    j["parent_dataset_id"] = meta.parent_dataset_id.toStdString();
    return j;
}

json HttpServer::buildPlotPayload(std::shared_ptr<AnnDataContainer> adata,
                                    const std::string& embedding_key,
                                    const std::string& color_by) {
    auto available = adata->getAvailableEmbeddings();
    if (available.empty()) {
        return json{{"error", "No two-dimensional embedding is available."}};
    }

    std::string emb = embedding_key.empty() ? adata->defaultEmbeddingKey() : embedding_key;
    if (emb.empty() || adata->obsm.find(emb) == adata->obsm.end()) {
        emb = available[0];
    }

    auto& coords = adata->obsm[emb];
    int n = static_cast<int>(coords.size());

    std::string col = color_by.empty() ? adata->defaultColorColumn() : color_by;

    json points;
    points["x"] = json::array();
    points["y"] = json::array();
    points["ids"] = json::array();

    for (int i = 0; i < n; i++) {
        points["x"].push_back(static_cast<double>(coords[i][0]));
        points["y"].push_back(static_cast<double>(coords[i][1]));
        points["ids"].push_back(i);
    }

    // Color values
    if (!col.empty() && adata->obs.count(col)) {
        auto& obs_col = adata->obs[col];
        points["color_values"] = json::array();
        if (obs_col.type == AnnDataContainer::ObsColumn::STRING) {
            for (const auto& v : obs_col.string_values) {
                points["color_values"].push_back(v);
            }
        } else if (obs_col.type == AnnDataContainer::ObsColumn::INT) {
            for (auto v : obs_col.int_values) {
                points["color_values"].push_back(v);
            }
        } else if (obs_col.type == AnnDataContainer::ObsColumn::FLOAT) {
            for (auto v : obs_col.float_values) {
                points["color_values"].push_back(v);
            }
        }
    }

    json j;
    j["embedding_key"] = emb;
    j["color_by"] = col;
    j["x_label"] = emb + "_1";
    j["y_label"] = emb + "_2";
    j["points"] = points;
    return j;
}

void HttpServer::start() {
    m_running = true;
    m_thread = std::make_unique<std::thread>([this]() {
        crow::SimpleApp app;

        // --- Health check ---
        CROW_ROUTE(app, "/api/health")([]() {
            return json_response(json{{"status", "ok"}}.dump());
        });

        // --- Upload h5ad file ---
        CROW_ROUTE(app, "/api/upload").methods(crow::HTTPMethod::Post)
        ([this](const crow::request& req) {
            try {
                // Frontend sends FormData with "file" field
                // For the Qt desktop app, we also accept JSON with "file_path"
                std::string file_path;

                // Try JSON body first (Qt file dialog path)
                try {
                    json body = json::parse(req.body);
                    if (body.contains("file_path")) {
                        file_path = body["file_path"].get<std::string>();
                    }
                } catch (...) {
                    // Not JSON - could be multipart form data from browser
                    // Parse multipart to find file content
                    // Simple approach: save the raw body as a temp h5ad file
                    // For now, look for file_path in the multipart boundaries
                    const std::string& raw = req.body;

                    // Check if it's multipart form data with a file
                    // Extract the h5ad binary content between boundaries
                    // This is a simplified parser - finds the HDF5 magic number
                    size_t hdf5_pos = raw.find("HDF");
                    if (hdf5_pos != std::string::npos && hdf5_pos > 0) {
                        // Find the actual start of HDF5 data (magic: \x89HDF\r\n\x1a\n)
                        size_t data_start = raw.rfind("\r\n\r\n", hdf5_pos);
                        if (data_start != std::string::npos) {
                            data_start += 4; // skip \r\n\r\n
                        } else {
                            data_start = hdf5_pos;
                        }

                        // Find end boundary
                        size_t data_end = raw.find("\r\n--", data_start);
                        std::string hdf5_data = (data_end != std::string::npos)
                            ? raw.substr(data_start, data_end - data_start)
                            : raw.substr(data_start);

                        // Write to temp file
                        QString temp_path = QDir::tempPath() + "/atlas_lens_upload_" +
                            QUuid::createUuid().toString(QUuid::WithoutBraces) + ".h5ad";
                        QFile temp_file(temp_path);
                        if (temp_file.open(QIODevice::WriteOnly)) {
                            temp_file.write(hdf5_data.data(), hdf5_data.size());
                            temp_file.close();
                            file_path = temp_path.toStdString();
                        }
                    }
                }

                if (file_path.empty()) {
                    return json_response(400, json{{"detail", "No file provided. Use file_path in JSON body or upload via FormData."}}.dump());
                }

                auto adata = H5adReader::read(file_path);
                if (!adata) {
                    return json_response(400, json{{"detail","Failed to read h5ad file."}}.dump());
                }

                adata->dataset_name = QFileInfo(QString::fromStdString(file_path)).fileName().toStdString();
                QString dataset_id = m_dataset_store.add(
                    QString::fromStdString(adata->dataset_name), adata);

                auto summary = buildDatasetSummary(dataset_id, adata);
                json response;
                response["summary"] = summary;
                if (!adata->defaultEmbeddingKey().empty()) {
                    response["plot"] = buildPlotPayload(adata, adata->defaultEmbeddingKey(), adata->defaultColorColumn());
                } else {
                    response["plot"] = nullptr;
                }
                return json_response(response.dump());
            } catch (const std::exception& e) {
                return json_response(400, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Load sample dataset ---
        CROW_ROUTE(app, "/api/load-sample").methods(crow::HTTPMethod::Post)
        ([this](const crow::request& req) {
            try {
                // Find sample h5ad in sample_data directory
                QString sample_dir = QDir::currentPath() + "/sample_data";
                QStringList filters;
                filters << "*.h5ad";
                auto entries = QDir(sample_dir).entryList(filters);
                if (entries.isEmpty()) {
                    return json_response(404, json{{"detail","No sample h5ad files found."}}.dump());
                }

                QString sample_path = sample_dir + "/" + entries[0];
                auto adata = H5adReader::read(sample_path.toStdString());
                if (!adata) {
                    return json_response(400, json{{"detail","Failed to read sample file."}}.dump());
                }

                adata->dataset_name = entries[0].toStdString();
                QString dataset_id = m_dataset_store.add(entries[0], adata);

                auto summary = buildDatasetSummary(dataset_id, adata);
                json response;
                response["summary"] = summary;
                if (!adata->defaultEmbeddingKey().empty()) {
                    response["plot"] = buildPlotPayload(adata, adata->defaultEmbeddingKey(), adata->defaultColorColumn());
                }
                return json_response(response.dump());
            } catch (const std::exception& e) {
                return json_response(400, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Get plot data ---
        CROW_ROUTE(app, "/api/datasets/<string>/plot").methods(crow::HTTPMethod::Post)
        ([this](const crow::request& req, std::string dataset_id) {
            try {
                auto adata = m_dataset_store.get(QString::fromStdString(dataset_id));
                json body;
                try { body = json::parse(req.body); } catch (...) {}

                std::string embedding_key = body.value("embedding_key", "");
                std::string color_by = body.value("color_by", "");

                embedding_key = resolveEmbeddingKey(adata, embedding_key);
                color_by = resolveColorColumn(adata, color_by);

                auto plot = buildPlotPayload(adata, embedding_key, color_by);
                return json_response(plot.dump());
            } catch (const std::exception& e) {
                return json_response(400, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Compute UMAP (not supported in C++, requires scanpy) ---
        CROW_ROUTE(app, "/api/datasets/<string>/compute-umap").methods(crow::HTTPMethod::Post)
        ([this](const crow::request& req, std::string dataset_id) {
            try {
                auto adata = m_dataset_store.get(QString::fromStdString(dataset_id));
                auto available = adata->getAvailableEmbeddings();
                if (!available.empty()) {
                    // Already have embeddings, just return current data
                    auto summary = buildDatasetSummary(QString::fromStdString(dataset_id), adata);
                    json response;
                    response["summary"] = summary;
                    response["plot"] = buildPlotPayload(adata, adata->defaultEmbeddingKey(), adata->defaultColorColumn());
                    return json_response(response.dump());
                }
                return json_response(400, json{{"detail", "UMAP computation requires scanpy (Python). This C++ build does not support compute-umap. Please use a dataset that already contains UMAP embeddings."}}.dump());
            } catch (const std::exception& e) {
                return json_response(400, json{{"detail", e.what()}}.dump());
            }
        });

        // --- Download artifact ---
        CROW_ROUTE(app, "/api/analysis-jobs/<string>/download/<string>")
        ([this](std::string job_id, std::string artifact) {
            try {
                auto job = m_job_store.get(QString::fromStdString(job_id));
                if (artifact == "mapping" && !job.result_info.value("mapping_path", "").empty()) {
                    std::string mapping_path = job.result_info.value("mapping_path", "");
                    QFile f(QString::fromStdString(mapping_path));
                    if (f.open(QIODevice::ReadOnly)) {
                        QByteArray data = f.readAll();
                        crow::response resp(std::string(data.constData(), data.size()));
                        resp.set_header("Content-Type", "application/json");
                        return resp;
                    }
                }
                return crow::response(404, "Artifact not found");
            } catch (const std::exception& e) {
                return crow::response(404, "Job not found");
            }
        });

        // --- Get dataset summary ---
        CROW_ROUTE(app, "/api/datasets/<string>/summary")
        ([this](std::string dataset_id) {
            try {
                auto adata = m_dataset_store.get(QString::fromStdString(dataset_id));
                auto summary = buildDatasetSummary(QString::fromStdString(dataset_id), adata);
                return json_response(summary.dump());
            } catch (const std::exception& e) {
                return json_response(404, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Create analysis job ---
        CROW_ROUTE(app, "/api/datasets/<string>/analysis-jobs").methods(crow::HTTPMethod::Post)
        ([this](const crow::request& req, std::string dataset_id) {
            try {
                auto adata = m_dataset_store.get(QString::fromStdString(dataset_id));
                json body = json::parse(req.body);

                std::string analysis_type = body["analysis_type"];

                QString job_id = QUuid::createUuid().toString(QUuid::WithoutBraces);
                QString job_dir = QDir::tempPath() + "/atlas_lens_jobs/" + job_id;
                QDir().mkpath(job_dir);

                AnalysisJob job;
                job.job_id = job_id;
                job.dataset_id = QString::fromStdString(dataset_id);
                job.analysis_type = QString::fromStdString(analysis_type);
                job.status = "running";
                job.message = "Running analysis...";
                job.progress = 0.25;
                job.job_dir = job_dir;
                job.created_at = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
                job.updated_at = job.created_at;

                if (analysis_type == "lasso_view") {
                    // Run LassoView synchronously for now
                    auto selected_ids = body["selected_ids"].get<std::vector<int>>();
                    std::string obs_col = body.value("obs_col", "");
                    bool do_correct = body.value("do_correct", true);

                    if (obs_col.empty()) obs_col = adata->defaultColorColumn();
                    if (obs_col.empty() || adata->obs.find(obs_col) == adata->obs.end()) {
                        return json_response(400, json{{"detail","LassoView requires a valid obs_col."}}.dump());
                    }

                    if (adata->obsp.find("connectivities") == adata->obsp.end()) {
                        return json_response(400, json{{"detail","No connectivities matrix found. Compute neighbors first."}}.dump());
                    }

                    // Get obs codes
                    auto& col = adata->obs[obs_col];
                    std::vector<int> obs_codes;
                    if (col.is_categorical) {
                        obs_codes = col.cat_codes;
                    } else if (col.type == AnnDataContainer::ObsColumn::INT) {
                        obs_codes = col.int_values;
                    } else {
                        // Map strings to ints
                        std::map<std::string, int> mapping;
                        for (const auto& v : col.string_values) {
                            if (mapping.find(v) == mapping.end()) {
                                mapping[v] = static_cast<int>(mapping.size());
                            }
                            obs_codes.push_back(mapping[v]);
                        }
                    }

                    int n_categories = *std::max_element(obs_codes.begin(), obs_codes.end()) + 1;
                    auto expanded_ids = LassoView::run(
                        adata->obsp["connectivities"],
                        obs_codes, selected_ids, n_categories, do_correct);

                    // Create result adata with lasso_view_status column
                    auto result_adata = std::make_shared<AnnDataContainer>(*adata);
                    result_adata->dataset_name = adata->dataset_name + " (lasso view)";

                    // Add lasso_view_status column
                    AnnDataContainer::ObsColumn status_col;
                    status_col.type = AnnDataContainer::ObsColumn::STRING;
                    status_col.string_values.resize(adata->n_obs, "unselected");
                    std::set<int> selected_set(selected_ids.begin(), selected_ids.end());
                    std::set<int> expanded_set(expanded_ids.begin(), expanded_ids.end());
                    for (int id : expanded_ids) {
                        if (selected_set.count(id)) {
                            status_col.string_values[id] = "seed";
                        } else {
                            status_col.string_values[id] = "propagated";
                        }
                    }
                    result_adata->obs["lasso_view_status"] = std::move(status_col);
                    result_adata->obs_column_order.push_back("lasso_view_status");

                    QString result_id = m_dataset_store.add(
                        QString::fromStdString(result_adata->dataset_name),
                        result_adata,
                        QString::fromStdString(dataset_id),
                        "lasso_view", true);

                    job.status = "completed";
                    job.progress = 1.0;
                    job.message = "Analysis completed.";
                    job.result_dataset_id = result_id;
                    job.result_info = json{
                        {"dataset_name", result_adata->dataset_name},
                        {"analysis_type", "lasso_view"},
                        {"preferred_embedding", adata->defaultEmbeddingKey()},
                        {"preferred_color_by", "lasso_view_status"},
                        {"expanded_ids", expanded_ids},
                        {"seed_ids", selected_ids}
                    };

                } else if (analysis_type == "downsample") {
                    std::string embedding_key = body.value("embedding_key", "");
                    embedding_key = resolveEmbeddingKey(adata, embedding_key);
                    if (embedding_key.empty()) {
                        return json_response(400, json{{"detail","Downsample requires an embedding."}}.dump());
                    }

                    double sample_rate = body.value("sample_rate", 0.1);
                    double uniform_rate = body.value("uniform_rate", 0.5);
                    double leiden_res = body.value("leiden_resolution", 1.0);
                    std::string cluster_key = body.value("cluster_key", "leiden");

                    auto ds_result = Downsample::run(adata, sample_rate, leiden_res, uniform_rate, cluster_key, embedding_key);

                    // Save nearest_ids mapping
                    json mapping;
                    mapping["nearest_downsampled_local_id"] = ds_result.nearest_ids;
                    std::vector<int> orig_idx;
                    if (ds_result.sampled_adata->obs.count("orig_idx")) {
                        orig_idx = ds_result.sampled_adata->obs["orig_idx"].int_values;
                    }
                    mapping["downsampled_orig_idx"] = orig_idx;
                    JsonUtils::writeJsonFile((job_dir + "/mapping.json").toStdString(), mapping);

                    QString result_id = m_dataset_store.add(
                        QString::fromStdString(ds_result.sampled_adata->dataset_name),
                        ds_result.sampled_adata,
                        QString::fromStdString(dataset_id),
                        "downsample", true);

                    job.status = "completed";
                    job.progress = 1.0;
                    job.message = "Analysis completed.";
                    job.result_dataset_id = result_id;
                    job.interactive_kind = "downsample";
                    job.result_info = json{
                        {"dataset_name", ds_result.sampled_adata->dataset_name},
                        {"analysis_type", "downsample"},
                        {"mapping_path", (job_dir + "/mapping.json").toStdString()},
                        {"preferred_embedding", embedding_key},
                        {"preferred_color_by", cluster_key},
                        {"interactive_kind", "downsample"}
                    };
                }

                m_job_store.create(job);
                return json_response(m_job_store.snapshot(job_id).dump());

            } catch (const std::exception& e) {
                return json_response(400, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Get analysis job status ---
        CROW_ROUTE(app, "/api/analysis-jobs/<string>")
        ([this](std::string job_id) {
            try {
                auto snap = m_job_store.snapshot(QString::fromStdString(job_id));
                // If job has a result, include summary and plot
                if (snap.contains("result_dataset_id") && !snap["result_dataset_id"].is_null()) {
                    auto result_id = snap["result_dataset_id"].get<std::string>();
                    try {
                        auto result_adata = m_dataset_store.get(QString::fromStdString(result_id));
                        snap["result_summary"] = buildDatasetSummary(QString::fromStdString(result_id), result_adata);
                        std::string emb = snap.value("result_info", json{})
                            .value("preferred_embedding", result_adata->defaultEmbeddingKey());
                        std::string col = snap.value("result_info", json{})
                            .value("preferred_color_by", result_adata->defaultColorColumn());
                        snap["result_plot"] = buildPlotPayload(result_adata, emb, col);
                    } catch (...) {}
                }
                return json_response(snap.dump());
            } catch (const std::exception& e) {
                return json_response(404, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Recover selection from downsampled ---
        CROW_ROUTE(app, "/api/analysis-jobs/<string>/recover-selection").methods(crow::HTTPMethod::Post)
        ([this](const crow::request& req, std::string job_id) {
            try {
                json body = json::parse(req.body);
                auto selected_ids = body["ids"].get<std::vector<int>>();

                auto job = m_job_store.get(QString::fromStdString(job_id));
                if (job.analysis_type != "downsample") {
                    return json_response(400, json{{"detail","Only downsample jobs support selection recovery."}}.dump());
                }

                auto mapping_path = job.result_info.value("mapping_path", "");
                auto mapping = JsonUtils::readJsonFile(mapping_path);
                auto nearest_ids = mapping["nearest_downsampled_local_id"].get<std::vector<int>>();

                auto recovered = Downsample::recoverSelection(selected_ids, nearest_ids);
                return json_response(json{{"ids", recovered}, {"count", recovered.size()}}.dump());
            } catch (const std::exception& e) {
                return json_response(400, json{{"detail",e.what()}}.dump());
            }
        });

        // --- Export selection ---
        CROW_ROUTE(app, "/api/datasets/<string>/export-selection")
        ([this](const crow::request& req, std::string dataset_id) {
            // Simple: parse ids from query param
            std::string ids_str = req.url_params.get("ids") ? std::string(req.url_params.get("ids")) : "";
            json ids_json;
            try {
                ids_json = json::parse(ids_str);
            } catch (...) {
                // Try comma-separated
                std::vector<int> ids;
                std::istringstream iss(ids_str);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    try { ids.push_back(std::stoi(token)); } catch (...) {}
                }
                ids_json = ids;
            }
            return json_response(json{{"ids", ids_json}}.dump());
        });

        // --- Serve frontend static files ---
        CROW_ROUTE(app, "/static/<path>")
        ([](std::string path) {
            QString file_path = QDir::currentPath() + "/frontend/" + QString::fromStdString(path);
            QFile f(file_path);
            if (f.open(QIODevice::ReadOnly)) {
                QByteArray data = f.readAll();
                std::string content_type = "text/plain";
                if (str_ends_with(path, ".html")) content_type = "text/html";
                else if (str_ends_with(path, ".js")) content_type = "application/javascript";
                else if (str_ends_with(path, ".jsx")) content_type = "application/javascript";
                else if (str_ends_with(path, ".css")) content_type = "text/css";
                else if (str_ends_with(path, ".json")) content_type = "application/json";

                crow::response resp(std::string(data.constData(), data.size()));
                resp.set_header("Content-Type", content_type);
                return resp;
            }
            return crow::response(404, "Not found");
        });

        // --- Serve index.html at root ---
        CROW_ROUTE(app, "/")([]() {
            QString file_path = QDir::currentPath() + "/frontend/index.html";
            QFile f(file_path);
            if (f.open(QIODevice::ReadOnly)) {
                QByteArray data = f.readAll();
                crow::response resp(std::string(data.constData(), data.size()));
                resp.set_header("Content-Type", "text/html");
                return resp;
            }
            return crow::response(404, "Frontend not found");
        });

        app.port(m_port).concurrency(4).run();
    });
}

void HttpServer::stop() {
    if (m_running) {
        m_running = false;
        // Crow doesn't have a clean stop mechanism for SimpleApp
        // In production, use app.stop() with a signal
    }
    if (m_thread && m_thread->joinable()) {
        m_thread->detach();
    }
}
