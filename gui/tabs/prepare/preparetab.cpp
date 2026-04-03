#include "preparetab.h"
#include "ui_preparetab.h"

PrepareTab::PrepareTab(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::PrepareTab)
{
    ui->setupUi(this);

    //ui->visual3DWidget->loadSTL("C:/Users/canca/Downloads/pumpkin.stl");
    ui->visual3DWidget->setModelVisible(true);
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

void PrepareTab::displaySTLInViewer(const QString &filePath)
{
    if (!filePath.isEmpty() && ui->visual3DWidget) {
        ui->visual3DWidget->addSTLModel(filePath);
        qDebug() << "[PrepareTab] STL sent to ViewerWidget:" << filePath;
    }
}

void PrepareTab::rotateModel(int x, int y, int z)
{
    ui->visual3DWidget->rotateModel(x, y, z);
}
