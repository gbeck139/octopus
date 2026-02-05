#include "setupwizard.h"
#include "welcomepage.h"

#include <QKeyEvent>

SetupWizard::SetupWizard(QWidget *parent)
    : QWizard(parent)
{

    setWindowTitle("Setup Wizard");

    addPage(new WelcomePage);

    prusaSlicerPage = new PrusaSlicerPage;
    addPage(prusaSlicerPage);

    profilePage = new ProfilePage;
    addPage(profilePage);

    connect(profilePage, &ProfilePage::printerTypeSelected, this, &SetupWizard::printerTypeSelected);
}

void SetupWizard::accept()
{
    emit setupCompleted();
    QWizard::accept();
}


void SetupWizard::setFirstRunMode(bool enabled)
{
    isFirstRun = enabled;

    setOption(QWizard::NoCancelButton, enabled);
    //setOption(QWizard::DisabledBackButtonOnLastPage, enabled);

    if (enabled) {
        setWindowFlags(windowFlags() & ~Qt::WindowCloseButtonHint);
    }
}

void SetupWizard::setAvailablePrinters(const QList<PrinterViewData> &printers)
{
    profilePage->setAvailablePrinters(printers);
}

void SetupWizard::closeEvent(QCloseEvent *event)
{
    if (isFirstRun) {
        event->ignore();
        return;
    }
    QWizard::closeEvent(event);
}

void SetupWizard::keyPressEvent(QKeyEvent *event)
{
    if (isFirstRun && event->key() == Qt::Key_Escape) {
        event->accept();
        return;
    }
    QWizard::keyPressEvent(event);
}
