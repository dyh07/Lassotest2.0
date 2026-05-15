#include "data_store.h"
#include "h5ad_reader.h"
#include <QUuid>
#include <QDir>

static QString isoNow() {
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

QString DatasetStore::add(const QString& name, std::shared_ptr<AnnDataContainer> adata,
                           const QString& parent_id, const QString& analysis_type, bool is_derived) {
    QMutexLocker locker(&m_mutex);
    QString id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    DatasetRecord rec;
    rec.dataset_id = id;
    rec.dataset_name = name;
    rec.source_path = QString::fromStdString(adata->source_h5ad_path);
    rec.parent_dataset_id = parent_id;
    rec.analysis_type = analysis_type;
    rec.is_derived = is_derived;
    rec.created_at = isoNow();
    m_datasets[id] = adata;
    m_meta[id] = rec;
    return id;
}

QString DatasetStore::addFromPath(const QString& name, const QString& h5ad_path,
                                   const QString& parent_id, const QString& analysis_type, bool is_derived) {
    auto adata = H5adReader::read(h5ad_path.toStdString());
    if (!adata) {
        throw std::runtime_error("Failed to read h5ad file: " + h5ad_path.toStdString());
    }
    adata->dataset_name = name.toStdString();
    return add(name, adata, parent_id, analysis_type, is_derived);
}

std::shared_ptr<AnnDataContainer> DatasetStore::get(const QString& dataset_id) {
    QMutexLocker locker(&m_mutex);
    auto it = m_datasets.find(dataset_id);
    if (it == m_datasets.end()) {
        throw std::runtime_error("Dataset not found: " + dataset_id.toStdString());
    }
    return it.value();
}

DatasetRecord DatasetStore::getMeta(const QString& dataset_id) {
    QMutexLocker locker(&m_mutex);
    auto it = m_meta.find(dataset_id);
    if (it == m_meta.end()) {
        throw std::runtime_error("Dataset not found: " + dataset_id.toStdString());
    }
    return it.value();
}

QString DatasetStore::getName(const QString& dataset_id) {
    return getMeta(dataset_id).dataset_name;
}

QString DatasetStore::getSourcePath(const QString& dataset_id) {
    return getMeta(dataset_id).source_path;
}

void JobStore::create(const AnalysisJob& job) {
    QMutexLocker locker(&m_mutex);
    m_jobs[job.job_id] = job;
}

AnalysisJob JobStore::get(const QString& job_id) {
    QMutexLocker locker(&m_mutex);
    auto it = m_jobs.find(job_id);
    if (it == m_jobs.end()) {
        throw std::runtime_error("Job not found: " + job_id.toStdString());
    }
    return it.value();
}

void JobStore::update(const QString& job_id, const QString& status, const QString& message,
                       double progress, const QString& error) {
    QMutexLocker locker(&m_mutex);
    auto it = m_jobs.find(job_id);
    if (it == m_jobs.end()) return;
    auto& job = it.value();
    job.status = status;
    job.message = message;
    job.progress = progress;
    if (!error.isEmpty()) job.error = error;
    job.updated_at = isoNow();
}

void JobStore::setCompleted(const QString& job_id, const QString& result_dataset_id, const json& result_info) {
    QMutexLocker locker(&m_mutex);
    auto it = m_jobs.find(job_id);
    if (it == m_jobs.end()) return;
    auto& job = it.value();
    job.status = "completed";
    job.message = "Analysis completed.";
    job.progress = 1.0;
    job.result_dataset_id = result_dataset_id;
    job.result_info = result_info;
    job.updated_at = isoNow();
}

void JobStore::setFailed(const QString& job_id, const QString& error) {
    QMutexLocker locker(&m_mutex);
    auto it = m_jobs.find(job_id);
    if (it == m_jobs.end()) return;
    auto& job = it.value();
    job.status = "failed";
    job.message = error;
    job.error = error;
    job.updated_at = isoNow();
}

json JobStore::snapshot(const QString& job_id) {
    auto job = get(job_id);
    json j;
    j["job_id"] = job.job_id.toStdString();
    j["dataset_id"] = job.dataset_id.toStdString();
    j["analysis_type"] = job.analysis_type.toStdString();
    j["status"] = job.status.toStdString();
    j["message"] = job.message.toStdString();
    j["progress"] = job.progress;
    j["job_dir"] = job.job_dir.toStdString();
    j["created_at"] = job.created_at.toStdString();
    j["updated_at"] = job.updated_at.toStdString();
    if (!job.result_dataset_id.isEmpty())
        j["result_dataset_id"] = job.result_dataset_id.toStdString();
    if (!job.interactive_kind.isEmpty())
        j["interactive_kind"] = job.interactive_kind.toStdString();
    if (!job.error.isEmpty())
        j["error"] = job.error.toStdString();
    if (!job.result_info.is_null())
        j["result_info"] = job.result_info;
    return j;
}
