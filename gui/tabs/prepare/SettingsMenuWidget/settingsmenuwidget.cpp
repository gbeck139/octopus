#include "settingsmenuwidget.h"
#include "ui_settingsmenuwidget.h"

SettingsMenuWidget::SettingsMenuWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::SettingsMenuWidget)
{
    ui->setupUi(this);

    // UI Connects
    connect(ui->printerCombo, &QComboBox::currentIndexChanged, this, &SettingsMenuWidget::onPrinterSelected);

    // UI <-> UI Connects
    connect(ui->printerSettings, &QGroupBox::toggled, ui->printerContent, &QFrame::setVisible);
    connect(ui->filamentSettings, &QGroupBox::toggled, ui->filamentContent, &QFrame::setVisible);
}

SettingsMenuWidget::~SettingsMenuWidget()
{
    delete ui;
}

void SettingsMenuWidget::populatePrinterCombo(const QList<const PrinterProfile*> system, const QList<const PrinterProfile*> user, const QString &activePrinterId)
{
    QSignalBlocker blocker(ui->printerCombo);

    ui->printerCombo->clear();

    // System Presets Header
    ui->printerCombo->addItem("-- System Presets --");
    QModelIndex idx = ui->printerCombo->model()->index(0,0);
    ui->printerCombo->model()->setData(idx, 0, Qt::UserRole - 1);

    int selectedIndex = -1;

    // System Preset Printers
    for (const PrinterProfile* profile : system) {
        ui->printerCombo->addItem(profile->getDisplayName());
        int row = ui->printerCombo->count() - 1;

        ui->printerCombo->setItemData(row, profile->getId(), Qt::UserRole);

        qDebug() << "Comparing:" << profile->getId() << "vs" << activePrinterId;

        if (profile->getId() == activePrinterId)
            selectedIndex = row;
    }

    // User Presets Header
    ui->printerCombo->addItem("-- User Presets --");
    idx = ui->printerCombo->model()->index(ui->printerCombo->count() - 1, 0);
    ui->printerCombo->model()->setData(idx, 0, Qt::UserRole - 1);

    // User Preset Printers
    for (const PrinterProfile* profile : user) {
        ui->printerCombo->addItem(profile->getDisplayName());
        int row = ui->printerCombo->count() - 1;

        ui->printerCombo->setItemData(row, profile->getId(), Qt::UserRole);

        if (profile->getId() == activePrinterId)
            selectedIndex = row;
    }

    qDebug() << "current selected index:" << selectedIndex;

    // Set active printer as current index
    if (selectedIndex >= 0) {
        ui->printerCombo->setCurrentIndex(selectedIndex);
    }
}

void SettingsMenuWidget::onPrinterSelected(int index)
{
    QString printerId =
        ui->printerCombo->itemData(index, Qt::UserRole).toString();

    if (!printerId.isEmpty())
        qDebug() << "[SETTINGSMENUWIDGET] onPrinterSelected emited";
        emit printerSelected(printerId);
}
