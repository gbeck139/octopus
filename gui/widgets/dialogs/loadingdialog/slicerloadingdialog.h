#ifndef SLICERLOADINGDIALOG_H
#define SLICERLOADINGDIALOG_H

#include <QDialog>
#include <QTimer>

namespace Ui {
class SlicerLoadingDialog;
}

class SlicerLoadingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SlicerLoadingDialog(QWidget *parent = nullptr);
    ~SlicerLoadingDialog();

private:
    Ui::SlicerLoadingDialog *ui;
    int angle;
    QTimer *timer;
    QPixmap originalPixmap;

private:
    void rotateCat();
};

#endif // SLICERLOADINGDIALOG_H
