#include "preparetab.h"
#include "ui_preparetab.h"

PrepareTab::PrepareTab(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::PrepareTab)
{
    ui->setupUi(this);
}

PrepareTab::~PrepareTab()
{
    delete ui;
}
