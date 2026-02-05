#ifndef SETTINGSMENUWIDGET_H
#define SETTINGSMENUWIDGET_H

#include <QWidget>
#include "printerviewdata.h"

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
    void populatePrinterCombo(const QList<PrinterViewData>& system, const QList<PrinterViewData>& user, const QString &activePrinterId);
    void rebuildPrinterCombo(const QList<PrinterViewData>& system, const QList<PrinterViewData>& user);
    void refreshActivePrinterDisplay(const QString& activePrinterId);

private slots:
    void onPrinterSelected(int index);
    void onEditPrinterClicked();

private:
    Ui::SettingsMenuWidget *ui;
};

#endif // SETTINGSMENUWIDGET_H
