#ifndef SETUPWIZARD_H
#define SETUPWIZARD_H

#include <QWizard>
#include "profilepage.h"

class SetupWizard : public QWizard
{
    Q_OBJECT
public:
    explicit SetupWizard(QWidget *parent = nullptr);
    void setFirstRunMode(bool enabled);

protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void accept() override;
signals:
    void printerTypeSelected(QString printerId);
    void setupCompleted();
private:
    ProfilePage *profilePage;
    bool isFirstRun = false;
};

#endif // SETUPWIZARD_H
