#ifndef SETUPWIZARD_H
#define SETUPWIZARD_H

#include <QWizard>
#include "profilepage.h"

class SetupWizard : public QWizard
{
    Q_OBJECT
public:
    explicit SetupWizard(bool firstRun, QWidget *parent = nullptr);
    ProfilePage *profilePage;
protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
private:
    bool isFirstRun;
};

#endif // SETUPWIZARD_H
