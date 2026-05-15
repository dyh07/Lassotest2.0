#pragma once

#include "ann_data.h"
#include <QMutex>
#include <QMap>
#include <QString>
#include <QDateTime>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
//存储数据集元信息
struct DatasetRecord {
    QString dataset_id;
    QString dataset_name;
    QString source_path;
    QString parent_dataset_id;
    QString analysis_type;//分析方式
    bool is_derived = false;//是否为衍生是数据集
    QString created_at;
};

struct AnalysisJob {
    QString job_id;
    QString dataset_id;
    QString analysis_type;
    QString status;      // queued, running, completed, failed
    QString message;
    double progress = 0.0;
    QString job_dir;
    QString created_at;
    QString updated_at;
    QString result_dataset_id;
    QString interactive_kind;
    QString error;
    json result_info;
};
//管理所有已加载的数据集及其元数据
class DatasetStore {
public:
    QString add(const QString& name, std::shared_ptr<AnnDataContainer> adata,
                const QString& parent_id = "", const QString& analysis_type = "", bool is_derived = false);
    QString addFromPath(const QString& name, const QString& h5ad_path,
                        const QString& parent_id = "", const QString& analysis_type = "", bool is_derived = false);
    std::shared_ptr<AnnDataContainer> get(const QString& dataset_id);
    DatasetRecord getMeta(const QString& dataset_id);
    QString getName(const QString& dataset_id);
    QString getSourcePath(const QString& dataset_id);

private:
    QMap<QString, std::shared_ptr<AnnDataContainer>> m_datasets;
    QMap<QString, DatasetRecord> m_meta;
    QMutex m_mutex;
};

class JobStore {
public:
    void create(const AnalysisJob& job);
    AnalysisJob get(const QString& job_id);//返回作业的完整 AnalysisJob 对象
    void update(const QString& job_id, const QString& status, const QString& message,
                double progress, const QString& error = "");
    void setCompleted(const QString& job_id, const QString& result_dataset_id, const json& result_info);
    void setFailed(const QString& job_id, const QString& error);
    json snapshot(const QString& job_id);

private:
    QMap<QString, AnalysisJob> m_jobs;
    QMutex m_mutex;
};
