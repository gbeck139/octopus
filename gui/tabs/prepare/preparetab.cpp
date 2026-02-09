#include "preparetab.h"
#include "ui_preparetab.h"

PrepareTab::PrepareTab(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::PrepareTab)
{
    ui->setupUi(this);

    ui->visual3DWidget->loadSTL("C:/Users/canca/Downloads/3DBenchy_deformed.gcode");
}

PrepareTab::~PrepareTab()
{
    delete ui;
}

SettingsMenuWidget *PrepareTab::getSettingsMenu() const
{
    return ui->settingsMenuWidget;
}

void PrepareTab::onPrinterChanged(const QString &printerId)
{
    qDebug() << "[UI UPDATE] Active printer changed to:" << printerId;

    //TODO:
    // - update UI limits
    // - update build volume display
    // - update currently selected printer
}

