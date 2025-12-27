#include "setupwizard.h"
#include "welcomepage.h"

#include <QKeyEvent>

SetupWizard::SetupWizard(bool firstRun, QWidget *parent)
    : QWizard(parent),
    isFirstRun(firstRun)
{
    if (isFirstRun) {
        // Remove Cancel and close button
        setOption(QWizard::NoCancelButton, true);
        setWindowFlags(windowFlags() & ~Qt::WindowCloseButtonHint);
    }

    setWindowTitle("Setup Wizard");

    profilePage = new ProfilePage;

    addPage(new WelcomePage);
    addPage(profilePage);
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
