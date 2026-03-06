#include "slicerloadingdialog.h"
#include "ui_slicerloadingdialog.h"

SlicerLoadingDialog::SlicerLoadingDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::SlicerLoadingDialog)
{
    ui->setupUi(this);

    setWindowTitle("Slicing in progress...");
    setModal(true);
    setFixedSize(300, 300);

    originalPixmap.load(":/images/images/catLoading.jpg"); //resource cat image
    ui->catLabel->setPixmap(originalPixmap);

    angle = 0;

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &SlicerLoadingDialog::rotateCat);
    timer->start(30);
}

SlicerLoadingDialog::~SlicerLoadingDialog()
{
    delete ui;
}

void SlicerLoadingDialog::rotateCat()
{
    QTransform t;
    t.rotate(angle);
    ui->catLabel->setPixmap(originalPixmap.transformed(t, Qt::SmoothTransformation));
    angle += 5;
    if (angle >= 360) angle = 0;
}
