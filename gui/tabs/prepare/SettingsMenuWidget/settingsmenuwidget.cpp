#include "settingsmenuwidget.h"
#include "ui_settingsmenuwidget.h"

#include <QSignalBlocker>
#include <QDebug>

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
    connect(ui->editPrinterButton, &QPushButton::clicked, this, &SettingsMenuWidget::onEditPrinterClicked);
}

SettingsMenuWidget::~SettingsMenuWidget()
{
    delete ui;
}

void SettingsMenuWidget::populatePrinterCombo(const QList<PrinterViewData>& system, const QList<PrinterViewData>& user, const QString &activePrinterId)
{
    QSignalBlocker blocker(ui->printerCombo);
    ui->printerCombo->clear();

    int selectedIndex = -1;

    // System Presets Header
    ui->printerCombo->addItem("-- System Presets --");
    QModelIndex idx = ui->printerCombo->model()->index(0,0);
    ui->printerCombo->model()->setData(idx, 0, Qt::UserRole - 1);

    // System Preset Printers
    for (const auto& profile : system) {
        ui->printerCombo->addItem(profile.name);
        int row = ui->printerCombo->count() - 1;
        ui->printerCombo->setItemData(row, profile.id, Qt::UserRole);

        qDebug() << "[SETTINGSMENUWIDGET] Comparing:" << profile.id << "vs" << activePrinterId;

        if (profile.id == activePrinterId)
            selectedIndex = row;
    }

    // User Presets Header
    ui->printerCombo->addItem("-- User Presets --");
    idx = ui->printerCombo->model()->index(ui->printerCombo->count() - 1, 0);
    ui->printerCombo->model()->setData(idx, 0, Qt::UserRole - 1);

    // User Preset Printers
    for (const auto& profile : user) {
        ui->printerCombo->addItem(profile.name);
        int row = ui->printerCombo->count() - 1;

        ui->printerCombo->setItemData(row, profile.id, Qt::UserRole);

        if (profile.id == activePrinterId)
            selectedIndex = row;
    }

    qDebug() << "current selected index:" << selectedIndex;

    // Set active printer as current index
    if (selectedIndex >= 0) {
        ui->printerCombo->setCurrentIndex(selectedIndex);
    }
}

void SettingsMenuWidget::rebuildPrinterCombo(const QList<PrinterViewData>& system, const QList<PrinterViewData>& user)
{
    populatePrinterCombo(system, user, QString());
    // // Temporarily block signals so we don’t trigger printerSelected
    // QSignalBlocker blocker(ui->printerCombo);

    // ui->printerCombo->clear();

    // // System Presets Header
    // ui->printerCombo->addItem("-- System Presets --");
    // QModelIndex idx = ui->printerCombo->model()->index(0,0);
    // ui->printerCombo->model()->setData(idx, 0, Qt::UserRole - 1);

    // // System Preset Printers
    // for (const PrinterProfile* profile : system) {
    //     ui->printerCombo->addItem(profile->getDisplayName());
    //     int row = ui->printerCombo->count() - 1;
    //     ui->printerCombo->setItemData(row, profile->getId(), Qt::UserRole);
    // }

    // // User Presets Header
    // ui->printerCombo->addItem("-- User Presets --");
    // idx = ui->printerCombo->model()->index(ui->printerCombo->count() - 1, 0);
    // ui->printerCombo->model()->setData(idx, 0, Qt::UserRole - 1);

    // // User Preset Printers
    // for (const PrinterProfile* profile : user) {
    //     ui->printerCombo->addItem(profile->getDisplayName());
    //     int row = ui->printerCombo->count() - 1;
    //     ui->printerCombo->setItemData(row, profile->getId(), Qt::UserRole);
    // }
}

void SettingsMenuWidget::refreshActivePrinterDisplay(const QString &activePrinterId)
{
    QSignalBlocker blocker(ui->printerCombo);

    // Search for the index that matches the active printer ID
    int selectedIndex = -1;
    for (int i = 0; i < ui->printerCombo->count(); ++i) {
        QVariant id = ui->printerCombo->itemData(i, Qt::UserRole);
        if (id.isValid() && id.toString() == activePrinterId) {
            selectedIndex = i;
            break;
        }
    }

    if (selectedIndex >= 0) {
        ui->printerCombo->setCurrentIndex(selectedIndex);
    }

    // Optional: update other labels if you have them
    // e.g., ui->nozzleDiameterLabel->setText(activeProfile->getNozzleDiameter());
    // e.g., ui->limitsLabel->setText(activeProfile->getAxisLimitsString());

}

void SettingsMenuWidget::onPrinterSelected(int index)
{
    QString printerId =
        ui->printerCombo->itemData(index, Qt::UserRole).toString();

    if (!printerId.isEmpty())
        qDebug() << "[SETTINGSMENUWIDGET] onPrinterSelected emited";
    emit printerSelected(printerId);
}

void SettingsMenuWidget::onEditPrinterClicked()
{
    emit settingsMenuEditPrinterClicked();
}
