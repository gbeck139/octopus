#include "prusaslicerpage.h"
#include "ui_prusaslicerpage.h"

#include <QFileDialog>
#include <QMessageBox>

PrusaSlicerPage::PrusaSlicerPage(QWidget *parent)
    : QWizardPage(parent)
    , ui(new Ui::PrusaSlicerPage)
{
    ui->setupUi(this);

    setFinalPage(false);
    setCommitPage(false);

    setTitle("PrusaSlicer Install");
    setSubTitle("This application requires PrusaSlicer to be installed on your system.");

    //connect(ui->pathLineEdit, &QLineEdit::textChanged,
    //        this, &QWizardPage::completeChanged);
    connect(ui->browseButton, &QPushButton::clicked, this, &PrusaSlicerPage::browseButtonClicked);
}

PrusaSlicerPage::~PrusaSlicerPage()
{
    delete ui;
}

bool PrusaSlicerPage::isComplete() const
{
    return !ui->pathLabel->text().isEmpty()
    && QFile::exists(ui->pathLabel->text());
}

void PrusaSlicerPage::browseButtonClicked()
{
    QString prusaPath = QFileDialog::getOpenFileName(this, "Located PrusaSlicer Executable", "", "All Files (*)");

    if (!prusaPath.isEmpty()) {
        qDebug() << "PrusaSlicer path found";
    }

    bool isValidPath = isValidPrusaSlicer(prusaPath);

    if (!isValidPath) {
        qDebug() << "[ERROR] PrusaSlicer path is not valid";
        QMessageBox::warning(this, "Invalid Selection", "The selected file does not appear to be the PrusaSlicer executable.\n\n Please select the PrusaSlicer application or binary.");
        return;
    }

    ui->pathLabel->setText(prusaPath);
    emit completeChanged(); //WHAT???? i think imma need to give the prusapath to appconfig somehow?

}

bool PrusaSlicerPage::isValidPrusaSlicer(const QString &path)
{
    QFileInfo info(path);

    if (!info.exists() || !info.isFile())
        return false;

    if (path.endsWith(".app")) {
        QString internalExec = path + "/Contents/MacOS/PrusaSlicer";
        return QFileInfo(internalExec).isExecutable();
    }

    if (!info.isExecutable()) {
        return false;
    }

    QString fileName = info.fileName().toLower();

    // Accept common names across platforms
    //TODO: test this to make sure it works on windows, mac, and linux
    return fileName.contains("prusa");
}
