#include <QApplication>
#include <QWebEngineView>
#include <QWebEnginePage>
#include <QMainWindow>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QThread>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>

#include "http_server.h"
#include "data_store.h"
#include "h5ad_reader.h"
#include "json_utils.h"

static DatasetStore g_dataset_store;
static JobStore g_job_store;
static HttpServer* g_server = nullptr;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Atlas Lens - Single-Cell RNA-seq Visualizer");
        resize(1400, 900);

        // Create web view
        m_webView = new QWebEngineView(this);
        setCentralWidget(m_webView);

        // Create menu bar
        createMenus();

        // Create toolbar
        createToolBar();

        // Create status bar
        m_statusLabel = new QLabel("Ready");
        statusBar()->addWidget(m_statusLabel, 1);

        // Start HTTP server
        g_server = new HttpServer(g_dataset_store, g_job_store, 15114);
        g_server->start();

        // Load the frontend
        // Give the server a moment to start
        QTimer::singleShot(500, [this]() {
            m_webView->load(QUrl("http://localhost:15114/"));
            m_statusLabel->setText("Loading...");
        });

        // Connect load finished
        connect(m_webView, &QWebEngineView::loadFinished, [this](bool ok) {
            if (ok) {
                m_statusLabel->setText("Atlas Lens ready");
            } else {
                m_statusLabel->setText("Failed to load frontend");
            }
        });
    }

    ~MainWindow() {
        if (g_server) {
            g_server->stop();
            delete g_server;
        }
    }

private slots:
    void openFile() {
        QString path = QFileDialog::getOpenFileName(
            this, "Open h5ad File", QString(),
            "h5ad Files (*.h5ad);;All Files (*)");

        if (path.isEmpty()) return;

        m_statusLabel->setText("Loading " + QFileInfo(path).fileName() + "...");

        try {
            auto adata = H5adReader::read(path.toStdString());
            if (!adata) {
                QMessageBox::critical(this, "Error", "Failed to read h5ad file.");
                return;
            }

            adata->dataset_name = QFileInfo(path).fileName().toStdString();
            QString dataset_id = g_dataset_store.add(
                QString::fromStdString(adata->dataset_name), adata);

            m_statusLabel->setText(QString("Loaded %1 (%2 cells, %3 genes)")
                .arg(QString::fromStdString(adata->dataset_name))
                .arg(adata->n_obs)
                .arg(adata->n_vars));

            // Reload the web view to pick up the new dataset
            m_webView->page()->runJavaScript(
                QString("window.location.reload();")
            );

        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Error",
                QString("Failed to load file: %1").arg(e.what()));
        }
    }

    void openMatrixFolder() {
        QString folder = QFileDialog::getExistingDirectory(
            this, "Open Matrix Data Folder");

        if (folder.isEmpty()) return;

        m_statusLabel->setText("Loading matrix data...");

        try {
            auto adata = H5adReader::readFromMatrixFiles(folder.toStdString());
            if (!adata || adata->empty()) {
                QMessageBox::critical(this, "Error", "Failed to read matrix data.");
                return;
            }

            adata->dataset_name = QFileInfo(folder).fileName().toStdString();
            QString dataset_id = g_dataset_store.add(
                QString::fromStdString(adata->dataset_name), adata);

            m_statusLabel->setText(QString("Loaded %1 (%2 cells)")
                .arg(QString::fromStdString(adata->dataset_name))
                .arg(adata->n_obs));

            m_webView->page()->runJavaScript("window.location.reload();");

        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Error",
                QString("Failed to load matrix data: %1").arg(e.what()));
        }
    }

    void loadSample() {
        QString sample_dir = QDir::currentPath() + "/sample_data";
        QStringList filters;
        filters << "*.h5ad";
        auto entries = QDir(sample_dir).entryList(filters);

        if (entries.isEmpty()) {
            QMessageBox::information(this, "No Sample Data",
                "No sample h5ad files found in the sample_data directory.");
            return;
        }

        bool ok;
        QString selected = QInputDialog::getItem(this, "Load Sample",
            "Select a sample dataset:", entries, 0, false, &ok);
        if (!ok || selected.isEmpty()) return;

        m_statusLabel->setText("Loading sample...");

        try {
            // Trigger the API endpoint via JavaScript
            m_webView->page()->runJavaScript(
                QString("fetch('/api/load-sample?name=%1', {method:'POST'}).then(r=>r.json()).then(d=>window.ingestResponse&&window.ingestResponse(d));")
                    .arg(selected)
            );
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

private:
    void createMenus() {
        auto fileMenu = menuBar()->addMenu("&File");

        auto openAction = fileMenu->addAction("&Open h5ad...");
        openAction->setShortcut(QKeySequence::Open);
        connect(openAction, &QAction::triggered, this, &MainWindow::openFile);

        auto openMatrixAction = fileMenu->addAction("Open &Matrix Folder...");
        connect(openMatrixAction, &QAction::triggered, this, &MainWindow::openMatrixFolder);

        fileMenu->addSeparator();

        auto loadSampleAction = fileMenu->addAction("&Load Sample...");
        loadSampleAction->setShortcut(QKeySequence("Ctrl+L"));
        connect(loadSampleAction, &QAction::triggered, this, &MainWindow::loadSample);

        fileMenu->addSeparator();

        auto quitAction = fileMenu->addAction("&Quit");
        quitAction->setShortcut(QKeySequence::Quit);
        connect(quitAction, &QAction::triggered, this, &QWidget::close);
    }

    void createToolBar() {
        auto toolbar = addToolBar("Main");
        toolbar->setMovable(false);

        auto openBtn = toolbar->addAction("Open h5ad");
        connect(openBtn, &QAction::triggered, this, &MainWindow::openFile);

        auto matrixBtn = toolbar->addAction("Open Matrix");
        connect(matrixBtn, &QAction::triggered, this, &MainWindow::openMatrixFolder);

        toolbar->addSeparator();

        auto sampleBtn = toolbar->addAction("Load Sample");
        connect(sampleBtn, &QAction::triggered, this, &MainWindow::loadSample);
    }

    QWebEngineView* m_webView;
    QLabel* m_statusLabel;
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Set application info
    app.setApplicationName("Atlas Lens");
    app.setOrganizationName("Lasso");

    MainWindow window;
    window.show();

    return app.exec();
}

#include "main.moc"
