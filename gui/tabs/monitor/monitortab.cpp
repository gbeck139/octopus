#include "monitortab.h"
#include "ui_monitortab.h"

MonitorTab::MonitorTab(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MonitorTab)
{
    ui->setupUi(this);
}

MonitorTab::~MonitorTab()
{
    delete ui;
}
