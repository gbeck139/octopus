#ifndef PRINTERPROFILEDIALOG_H
#define PRINTERPROFILEDIALOG_H

#include <QDialog>
#include "printerprofile.h"

namespace Ui {
class printerProfileDialog;
}

class printerProfileDialog : public QDialog
{
    Q_OBJECT

public:
    explicit printerProfileDialog(const PrinterProfile* originalPrinter, QWidget *parent = nullptr);
    ~printerProfileDialog();

signals:
    void saveRequested(PrinterProfile* updated);
    void saveAsRequested(PrinterProfile* updated);

public slots:
    void onSaveClicked();
    void onSaveAsClicked();

private:
    Ui::printerProfileDialog *ui;
    PrinterProfile *printerCopy;

private:
    void setPrinter(const PrinterProfile* printer);
    void applyChangesToProfile(PrinterProfile* profile);
};

#endif // PRINTERPROFILEDIALOG_H
