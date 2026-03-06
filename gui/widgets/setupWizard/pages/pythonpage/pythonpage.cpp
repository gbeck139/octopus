#include "pythonpage.h"
#include "ui_pythonpage.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>

PythonPage::PythonPage(QWidget *parent)
    : QWizardPage(parent)
    , ui(new Ui::PythonPage)
{
    ui->setupUi(this);

    setFinalPage(false);
    setCommitPage(false);

    setTitle("Temporary Python Install");
    setSubTitle("This application (TEMPORARILY) requires Python to be installed on your system.");

    ui->label->setFocusPolicy(Qt::NoFocus);
    ui->label->setAttribute(Qt::WA_TransparentForMouseEvents, false);
    ui->label->setTextInteractionFlags(Qt::TextBrowserInteraction);
    ui->label->setOpenExternalLinks(true);


    connect(ui->browseButton, &QPushButton::clicked, this, &PythonPage::browseButtonClicked);
}

PythonPage::~PythonPage()
{
    delete ui;
}

bool PythonPage::isComplete() const
{
    return !ui->pathLineEdit->text().isEmpty()
    && QFile::exists(ui->pathLineEdit->text());
}

void PythonPage::browseButtonClicked()
{
    QString pythonPath = QFileDialog::getOpenFileName(this, "Locate Python", QDir::homePath(), "All Files (*)");

    if (pythonPath.isEmpty()) {
        return;
    }

    bool isValidPath = isValidPython(pythonPath);

    if (!isValidPath) {
        qDebug() << "[ERROR] Python path is not valid";
        QMessageBox::warning(this, "Invalid Selection", "The selected file does not appear to be Python.\n\n Please try selecting again.");
        return;
    }

    ui->pathLineEdit->setText(pythonPath);

    emit pythonPathSelected(pythonPath);
    emit completeChanged();

}

bool PythonPage::isValidPython(const QString &path)
{
    QProcess process;
    process.start(path, {"--version"});
    if (!process.waitForFinished(2000))  // wait 2 seconds
        return false;

    QString output = process.readAllStandardOutput();
    QString error = process.readAllStandardError();

    return output.contains("Python") || error.contains("Python");
}


