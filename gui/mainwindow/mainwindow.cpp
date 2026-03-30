#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "preparetab.h"
#include "printerprofiledialog.h"

#include <QFileDialog>
#include <QMessageBox>

#include <QDebug>
#include <QLabel>
#include <QWindow>
#include <QSlider>
#include <QCheckBox>

//TODO: move this to model3d model
struct ModelRotation {
    int x = 0;
    int y = 0;
    int z = 0;
};
ModelRotation currentRotation;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Model references
    appConfig = new AppConfig(this);
    model3D = new Model3D(this);
    profileManager = new ProfileManager(this);
    slicerRunner = new SlicerRunner(this);
    loadingDialog = new SlicerLoadingDialog(this);

    // Set Printer Profiles
    if (!appConfig->isFirstRun()) {
        profileManager->setActivePrinter(appConfig->getActivePrinterId());
    }

    //TESTING
    //profileManager->deleteUserPrinter("cylinderOne(1)");


    // Settings, populate profiles
    SettingsMenuWidget* settingsMenu = ui->prepareTabWidget->getSettingsMenu();
    // Populate printer combo
    settingsMenu->populatePrinterCombo(profileManager->getSystemPrintersForView(), profileManager->getUserPrintersForView(), profileManager->getActivePrinter());

    //Attach Slice and Print buttons to tab bar
    //TODO: make a qwidget class for this
    //ui->tabWidget->setCornerWidget(ui->slicePrintHWidget, Qt::TopRightCorner);


    //testing purposes~~~~~~~~~~~~~~~~~~~~~
    //appConfig->setFirstRunCompleted(false);

    // Show First Run Wizard
    if (appConfig->isFirstRun()) {
        SetupWizard* firstWizard = new SetupWizard(this);
        firstWizard->setFirstRunMode(true);

        QList<PrinterViewData> printers = profileManager->getSystemPrintersForView();
        firstWizard->setAvailablePrinters(printers);

        connectWizard(firstWizard);

        firstWizard->exec();
        firstWizard->deleteLater();
    }

    ui->exportButton->setEnabled(false);

    // UI Connects
    connect(ui->actionImport, &QAction::triggered, this, &MainWindow::onImportClicked);
    connect(ui->actionExport, &QAction::triggered, this, &MainWindow::onExportClicked);
    connect(ui->actionQuit, &QAction::triggered, this, &MainWindow::onQuitClicked);
    connect(ui->actionUndo, &QAction::triggered, this, &MainWindow::onUndoClicked);
    connect(ui->actionRedo, &QAction::triggered, this, &MainWindow::onRedoClicked);
    connect(ui->actionCut, &QAction::triggered, this, &MainWindow::onCutClicked);
    connect(ui->actionCopy, &QAction::triggered, this, &MainWindow::onCopyClicked);
    connect(ui->actionPaste, &QAction::triggered, this, &MainWindow::onPasteClicked);
    connect(ui->menuView, &QMenu::triggered, this, &MainWindow::onViewClicked);
    connect(ui->menuPreferences, &QMenu::triggered, this, &MainWindow::onPreferencesClicked);
    connect(ui->actionSetupWizard, &QAction::triggered, this, &MainWindow::onSetupWizardClicked);
    connect(ui->actionAbout, &QAction::triggered, this, &MainWindow::onAboutClicked);
    connect(ui->sliceButton, &QPushButton::clicked, this, &MainWindow::onSliceClicked);
    connect(ui->exportButton, &QPushButton::clicked, this, &MainWindow::onExportClicked);

    // Model Connects
    connect(settingsMenu, &SettingsMenuWidget::printerSelected, appConfig, &AppConfig::setActivePrinterId);
    connect(appConfig, &AppConfig::activePrinterChanged, profileManager, &ProfileManager::setActivePrinter);
    connect(profileManager, &ProfileManager::activePrinterChanged, settingsMenu, [=](const QString& id) {
        settingsMenu->populatePrinterCombo(profileManager->getSystemPrintersForView(), profileManager->getUserPrintersForView(), id);});

    connect(settingsMenu, &SettingsMenuWidget::settingsMenuEditPrinterClicked, this, &MainWindow::onSettingsMenuEditPrinterClicked);
    connect(profileManager, &ProfileManager::printersChanged, settingsMenu, [=]() {
        settingsMenu->rebuildPrinterCombo(profileManager->getSystemPrintersForView(), profileManager->getUserPrintersForView());});
    connect(profileManager, &ProfileManager::activePrinterDataChanged, settingsMenu, &SettingsMenuWidget::refreshActivePrinterDisplay);

    // Slicer Connects
    connect(slicerRunner, &SlicerRunner::sliceFinished, this, [](const QString& path){
        qDebug() << "[MAIN SLICER] Slice finished, G-code:" << path;
    });

    connect(slicerRunner, &SlicerRunner::sliceFailed, this, [](const QString& err){
        qDebug() << "[MAIN SLICER] Slice failed:" << err;
    });

    qDebug() << "[GUI] Current PrusaSlicer Path: " + appConfig->getPrusaSlicerPath();

    connect(ui->rotateXButton, &QPushButton::clicked, this, &MainWindow::onRotateXClicked);
    connect(ui->rotateYButton, &QPushButton::clicked, this, &MainWindow::onRotateYClicked);
    connect(ui->rotateZButton, &QPushButton::clicked, this, &MainWindow::onRotateZClicked);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onImportClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Import STL", "", "STL FILES (*.stl)");
    if (!filePath.isEmpty()) {
        model3D->loadModel(filePath);
        if (model3D->loadModel(filePath)) {
            ui->exportButton->setEnabled(false);
        }
        qDebug() << "STL file loaded successfully in GUI";
    } else {
        qDebug() << "STL not loaded into GUI";
        QMessageBox::warning(this, "Import failed", "Could not import 3D model file.");
    }
}

void MainWindow::onExportClicked()
{
    qDebug() << "Export Clicked";

    // Check model is loaded
    if (!model3D || !model3D->isLoaded()) {
        QMessageBox::warning(this, "No Model", "Please load a 3D model before exporting.");
        return;
    }

    if (lastGeneratedGcodePath.isEmpty() ||
        !QFile::exists(lastGeneratedGcodePath)) {
        QMessageBox::warning(this,
                             "No G-code",
                             "Please slice the model first.");
        return;
    }

    // Where to save g-code file
    QString savePath = QFileDialog::getSaveFileName(
        this,
        "Export G-code",
        "",
        "G-code Files (*.gcode)"
        );

    if (savePath.isEmpty())
        return;

    if (!savePath.endsWith(".gcode", Qt::CaseInsensitive))
        savePath += ".gcode";

    if (!QFile::copy(lastGeneratedGcodePath, savePath)) {
        QMessageBox::warning(this,
                             "Export Failed",
                             "Could not save file.");
        return;
    }

    QMessageBox::information(this,
                             "Export Complete",
                             "G-code saved successfully.");




    // TODO: SlicerRunner's exportGCode file
    //QString filePath = QFileDialog::getSaveFileName(this, "Export G-code", "", "G-code Files (*.gcode)");
    //if (!filePath.isEmpty()) {
    //    if (!slicerRunner->exportGCode(filePath)) {
    //        QMessageBox::warning(this, "Export Failed", "Could not export G-code file.");
    //    }
    //}
}

void MainWindow::onQuitClicked()
{
    // Provides an "Are you sure" message box if a model has been loaded in
    //TODO: if model has been exported/saved, then it's ok?
    //TODO: override closeEvent - just copy and paste, you can't connect together cuz quit() will make it a loop
    if (model3D->isLoaded()) {
        auto result = QMessageBox::question(this, "Quit", "Are you sure you want to quit?");
        if (result == QMessageBox::Yes) {
            QApplication::quit();
        }
    } else {
        QApplication::quit();
    }
}

void MainWindow::onUndoClicked()
{
    // TODO:
}

void MainWindow::onRedoClicked()
{
    // TODO:
}

void MainWindow::onCutClicked()
{
    // TODO:
}

void MainWindow::onCopyClicked()
{

}

void MainWindow::onPasteClicked()
{
    // TODO:
}

void MainWindow::onViewClicked()
{
    // TODO:
}

void MainWindow::onPreferencesClicked()
{
    // TODO:
    // Dialog ~ general, profiles, advanced
}

void MainWindow::onSetupWizardClicked()
{
    SetupWizard* setupWizard = new SetupWizard(this);
    setupWizard->setFirstRunMode(false);

    QList<PrinterViewData> printers = profileManager->getSystemPrintersForView();
    setupWizard->setAvailablePrinters(printers);

    connectWizard(setupWizard);

    setupWizard->exec();
    setupWizard->deleteLater();
}

void MainWindow::onAboutClicked()
{
    QMessageBox::about(this, "About", "Slicer App");
}

void MainWindow::onSetupCompleted()
{
    if (appConfig->isFirstRun()) {
        appConfig->markFirstRunCompleted();
    }
}

void MainWindow::connectWizard(SetupWizard *wizard)
{
    connect(wizard, &SetupWizard::printerTypeSelected, appConfig, &AppConfig::setActivePrinterId);
    connect(wizard, &SetupWizard::printerTypeSelected, profileManager, &ProfileManager::setActivePrinter);
    connect(wizard, &SetupWizard::prusaSlicerPathSelected, appConfig, &AppConfig::setPrusaSlicerPath);
    connect(wizard, &SetupWizard::setupCompleted, this, &MainWindow::onSetupCompleted);
}

void MainWindow::onSettingsMenuEditPrinterClicked()
{
    PrinterProfile* currentPrinter = profileManager->getActivePrinterProfile();

    auto* dialog = new printerProfileDialog(currentPrinter, this);

    connect(dialog, &printerProfileDialog::saveRequested, profileManager, &ProfileManager::updateUserPrinter);
    connect(dialog, &printerProfileDialog::saveAsRequested, profileManager, &ProfileManager::addUserPrinter);

    dialog->exec();
}

void MainWindow::onSliceClicked()
{
    if (!model3D || !model3D->isLoaded()) {
        QMessageBox::warning(this, "No Model", "Please load a 3D model before slicing.");
        return;
    }

    QString stlPath = model3D->getSourceFilePath();
    QString modelName = QFileInfo(stlPath).completeBaseName();

    qDebug() << "[MAIN] Triggering slice";

    loadingDialog->setWindowModality(Qt::ApplicationModal);
    loadingDialog->setWindowFlags(loadingDialog->windowFlags() & ~Qt::WindowCloseButtonHint);
    loadingDialog->show();

    QString slicerPath = QCoreApplication::applicationDirPath()
                         + "/slicerbundle/slicer_pipeline.exe";

    if (!QFile::exists(slicerPath)) {
        loadingDialog->hide();
        QMessageBox::critical(this, "Slicer Missing",
                              "The slicer executable could not be found at:\n" + slicerPath);
        return;
    }

    // ---- Run slicer process ----
    QProcess *proc = new QProcess(this);
    proc->setProcessChannelMode(QProcess::MergedChannels);

    connect(proc, &QProcess::readyReadStandardOutput, this, [proc]() {
        qDebug().noquote() << proc->readAllStandardOutput();
    });

    connect(proc, &QProcess::readyReadStandardError, this, [proc]() {
        qWarning().noquote() << proc->readAllStandardError();
    });

    connect(proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        loadingDialog->hide();
        QMessageBox::critical(this, "Process Error", "Failed to start the slicing process.");
    });

    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [=](int exitCode, QProcess::ExitStatus status) {

                loadingDialog->hide();

                if (status != QProcess::NormalExit || exitCode != 0) {
                    ui->exportButton->setEnabled(false);
                    QMessageBox::critical(this, "Slicing Failed", "Slicing process failed.");
                    proc->deleteLater();
                    return;
                }

                ui->exportButton->setEnabled(true);

                lastGeneratedGcodePath = QCoreApplication::applicationDirPath()
                                         + "/slicerbundle/output_gcode/" + modelName + "_reformed.gcode";

                // ---- Launch embedded Python visualizer ----
                QProcess *visualProcess = new QProcess(this);
                visualProcess->setProcessChannelMode(QProcess::SeparateChannels);
                visualProcess->setInputChannelMode(QProcess::ManagedInputChannel);

                QString visualPath = QCoreApplication::applicationDirPath()
                                     + "/slicerbundle/newest_visualizer_5.exe";

                if (!QFile::exists(visualPath)) {
                    qDebug() << "Visualizer executable not found at:" << visualPath;
                    proc->deleteLater();
                    return;
                }

                QStringList args;
                args << "--gcode" << lastGeneratedGcodePath;

                visualProcess->start(visualPath, args);
                if (!visualProcess->waitForStarted(3000)) {
                    qDebug() << "Failed to start visualizer process.";
                    proc->deleteLater();
                    return;
                }

                auto container_created = std::make_shared<bool>(false);

                connect(visualProcess, &QProcess::readyReadStandardOutput, this, [=]() mutable {
                    if (*container_created) return;

                    QByteArray data = visualProcess->readAllStandardOutput();
                    QString output = QString::fromUtf8(data).trimmed();
                    if (output.isEmpty()) return;

                    bool ok = false;
                    WId wid = output.toULongLong(&ok);
                    if (!ok) {
                        qDebug() << "Invalid WId output:" << output;
                        return;
                    }

                    qDebug() << "VALID WID:" << wid;
                    qDebug() << "Creating QWindow from WId...";
                    QWindow *foreignWindow = QWindow::fromWinId(wid);
                    QWidget *container = QWidget::createWindowContainer(foreignWindow, this);
                    container->setMinimumSize(800, 600);
                    container->setFocusPolicy(Qt::StrongFocus);
                    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

                    QLayoutItem *child;
                    while ((child = ui->previewTabLayout->takeAt(0)) != nullptr) {
                        delete child->widget();
                        delete child;
                    }

                    ui->previewTabLayout->addWidget(container);

                    // ---- UI CONTROLS ----
                    QSlider *historySlider = new QSlider(Qt::Horizontal, this);
                    historySlider->setMinimum(0);
                    historySlider->setMaximum(100);
                    historySlider->setValue(100);
                    ui->previewTabLayout->addWidget(historySlider);

                    QCheckBox *travelCheck = new QCheckBox("Travel", this);
                    ui->previewTabLayout->addWidget(travelCheck);

                    QCheckBox *densityCheck = new QCheckBox("Density", this);
                    ui->previewTabLayout->addWidget(densityCheck);

                    connect(historySlider, &QSlider::valueChanged, this, [=](int value){
                        QString cmd = QString("MOVE %1\n").arg(value);
                        qDebug() << "Sending:" << cmd;
                        visualProcess->write(cmd.toUtf8());
                    });

                    connect(travelCheck, &QCheckBox::toggled, this, [=](bool checked){
                        QString cmd = QString("TOGGLE_TRAVEL %1\n").arg(checked ? 1 : 0);
                        qDebug() << "Sending:" << cmd;
                        visualProcess->write(cmd.toUtf8());
                    });

                    connect(densityCheck, &QCheckBox::toggled, this, [=](bool checked){
                        QString cmd = QString("TOGGLE_DENSITY %1\n").arg(checked ? 1 : 0);
                        qDebug() << "Sending:" << cmd;
                        visualProcess->write(cmd.toUtf8());
                    });

                    ui->tabWidget->setCurrentIndex(ui->tabWidget->indexOf(ui->previewTab));
                    *container_created = true;
                });

                connect(visualProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                        this, [=](int code, QProcess::ExitStatus status) {
                            qDebug() << "Visualizer exited with code:" << code << "status:" << status;
                            visualProcess->deleteLater();
                        });

                proc->deleteLater();
            });

    // ---- Start slicer ----
    QStringList slicerArgs;
    slicerArgs << "--stl" << stlPath
               << "--model" << modelName
               << "--prusa" << appConfig->getPrusaSlicerPath();

    proc->start(slicerPath, slicerArgs);
}

void MainWindow::onRotateXClicked()
{

}

void MainWindow::onRotateYClicked()
{

}

void MainWindow::onRotateZClicked()
{

}

