#include "printerprofiledialog.h"
#include "ui_printerprofiledialog.h"

printerProfileDialog::printerProfileDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::printerProfileDialog)
{
    ui->setupUi(this);
}

printerProfileDialog::~printerProfileDialog()
{
    delete ui;
}
