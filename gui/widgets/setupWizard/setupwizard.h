#ifndef SETUPWIZARD_H
#define SETUPWIZARD_H

#include <QWizard>
#include "profilepage.h"
#include "prusaslicerpage.h"

class SetupWizard : public QWizard
{
    Q_OBJECT
public:
    explicit SetupWizard(QWidget *parent = nullptr);
    void setFirstRunMode(bool enabled);
    void setAvailablePrinters(const QList<PrinterViewData>& printers);

protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void accept() override;
signals:
    void printerTypeSelected(QString printerId);
    void setupCompleted();
private:
    ProfilePage *profilePage;
    PrusaSlicerPage *prusaSlicerPage;
    bool isFirstRun = false;
};

#endif // SETUPWIZARD_H
