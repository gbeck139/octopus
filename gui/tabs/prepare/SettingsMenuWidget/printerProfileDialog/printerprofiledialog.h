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
    explicit printerProfileDialog(const PrinterProfile* original, QWidget *parent = nullptr);
    ~printerProfileDialog();

signals:
    void saveRequested(PrinterProfile* updated);
    void saveAsRequested(PrinterProfile* updated);

private:
    Ui::printerProfileDialog *ui;
    PrinterProfile *printerCopy;
};

#endif // PRINTERPROFILEDIALOG_H
