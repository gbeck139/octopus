#include "printerprofiledialog.h"
#include "ui_printerprofiledialog.h"

printerProfileDialog::printerProfileDialog(const PrinterProfile *originalPrinter, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::printerProfileDialog)
{
    ui->setupUi(this);
    setWindowTitle("Edit Printer Profile");

    printerCopy = originalPrinter->clone();

    if (originalPrinter->isSystemProfile()) {
        ui->saveButton->setDisabled(true);
    }

    // Populate field with current printer data
    setPrinter(printerCopy);

    // UI Connects
    connect(ui->saveButton, &QPushButton::clicked, this, &printerProfileDialog::onSaveClicked);
    connect(ui->saveAsButton, &QPushButton::clicked, this, &printerProfileDialog::onSaveAsClicked);
    connect(ui->cancelButton, &QPushButton::clicked, this, &QDialog::reject);


}

printerProfileDialog::~printerProfileDialog()
{
    delete ui;
}

void printerProfileDialog::onSaveClicked()
{
    applyChangesToProfile(printerCopy); // copy values from spinboxes to 'printer' object
    emit saveRequested(printerCopy);
    accept();
}

void printerProfileDialog::onSaveAsClicked()
{
    PrinterProfile* newProfile = printerCopy->clone();

    applyChangesToProfile(newProfile); //copy values from spinboxes to a new 'printerprofile' object

    emit saveAsRequested(newProfile);
    emit saveRequested(newProfile);
    accept();
}

void printerProfileDialog::setPrinter(const PrinterProfile *printer)
{
    ui->nameLineEdit->setText(printer->getDisplayName());
    ui->nozzleDiameterDoubleSpinBox->setValue(printer->getNozzleDiameter());
    ui->xMinDoubleSpinBox->setValue(printer->getXMin());
    ui->xMaxDoubleSpinBox->setValue(printer->getXMax());
    ui->yMinDoubleSpinBox->setValue(printer->getYMin());
    ui->yMaxDoubleSpinBox->setValue(printer->getYMax());
    ui->zMinDoubleSpinBox->setValue(printer->getZMin());
    ui->zMaxDoubleSpinBox->setValue(printer->getZMax());
    ui->rotAxisMinDoubleSpinBox->setValue(printer->getRotMin());
    ui->rotAxisMaxDoubleSpinBox->setValue(printer->getRotMax());
}

void printerProfileDialog::applyChangesToProfile(PrinterProfile* profile)
{
    profile->setDisplayName(ui->nameLineEdit->text());
    profile->setNozzleDiameter(ui->nozzleDiameterDoubleSpinBox->value());
    profile->setAxisLimits(
        ui->xMinDoubleSpinBox->value(),
        ui->xMaxDoubleSpinBox->value(),
        ui->yMinDoubleSpinBox->value(),
        ui->yMaxDoubleSpinBox->value(),
        ui->zMinDoubleSpinBox->value(),
        ui->zMaxDoubleSpinBox->value(),
        ui->rotAxisMinDoubleSpinBox->value(),
        ui->rotAxisMaxDoubleSpinBox->value()
        );
    profile->setBuildVolume(
        ui->xMaxDoubleSpinBox->value(),
        ui->yMaxDoubleSpinBox->value(),
        ui->zMaxDoubleSpinBox->value()
        );
}
