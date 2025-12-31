#ifndef SETTINGSMENUWIDGET_H
#define SETTINGSMENUWIDGET_H

#include <QWidget>
#include "printerprofile.h"

namespace Ui {
class SettingsMenuWidget;
}

class SettingsMenuWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SettingsMenuWidget(QWidget *parent = nullptr);
    ~SettingsMenuWidget();

signals:
    void printerSelected(const QString& printerId);
    void settingsMenuEditPrinterClicked();

public slots:
    void populatePrinterCombo(const QList<const PrinterProfile*> system, const QList<const PrinterProfile*> user, const QString &activePrinterId);
    void onPrinterSelected(int index);
    void onEditPrinterClicked();
    void rebuildPrinterCombo(const QList<const PrinterProfile*>& system, const QList<const PrinterProfile*>& user);
    void refreshActivePrinterDisplay(const QString& activePrinterId);

private:
    Ui::SettingsMenuWidget *ui;
};

#endif // SETTINGSMENUWIDGET_H
