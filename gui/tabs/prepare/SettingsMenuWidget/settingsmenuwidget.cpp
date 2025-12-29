#include "settingsmenuwidget.h"
#include "ui_settingsmenuwidget.h"

SettingsMenuWidget::SettingsMenuWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::SettingsMenuWidget)
{
    ui->setupUi(this);

    connect(ui->printerSettings, &QGroupBox::toggled, ui->printerContent, &QFrame::setVisible);
    connect(ui->filamentSettings, &QGroupBox::toggled, ui->filamentContent, &QFrame::setVisible);
}

SettingsMenuWidget::~SettingsMenuWidget()
{
    delete ui;
}
