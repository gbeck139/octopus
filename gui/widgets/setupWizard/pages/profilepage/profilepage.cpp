#include "profilepage.h"
#include "ui_profilepage.h"

ProfilePage::ProfilePage(QWidget *parent)
    : QWizardPage(parent)
    , ui(new Ui::ProfilePage)
{
    ui->setupUi(this);

    setFinalPage(true);

    setTitle("Default Profiles");
    setSubTitle("Choose printer type");

    // May have to change later when there is more printers.
    // (Needs data structure that holds system/user Printer Profile Data)

    // Images
    //QPixmap cylOnePix(":/images/images/blueCylinder.jpg");
    //int widthOne = ui->cylinderOneImage->width();
    //int hieghtOne = ui->cylinderOneImage->height();
    //ui->cylinderOneImage->setPixmap(cylOnePix.scaled(widthOne,hieghtOne,Qt::KeepAspectRatio));

    //QPixmap cylTwoPix(":/images/images/redCylinder.jpg");
    //int widthTwo = ui->cylinderTwoImage->width();
    //int heightTwo = ui->cylinderTwoImage->height();
    //ui->cylinderTwoImage->setPixmap(cylTwoPix.scaled(widthTwo,heightTwo,Qt::KeepAspectRatio));

    printerGroup = new QButtonGroup(this);
    //printerGroup->addButton(ui->cylinderOne);
    //printerGroup->addButton(ui->cylinderTwo);
    //printerGroup->addButton(ui->radioButton);

    // Set printer IDs
    //ui->cylinderOne->setProperty("printerId", "cylinderOne");
    //ui->cylinderTwo->setProperty("printerId", "cylinderTwo");
    //ui->radioButton->setProperty("printerId", "otherPrint");

    // Wizard re-checks completeness when selection changes
    connect(printerGroup, &QButtonGroup::buttonClicked,
            this, &QWizardPage::completeChanged);

}

ProfilePage::~ProfilePage()
{
    delete ui;
}

bool ProfilePage::isComplete() const
{
    return printerGroup->checkedId() != -1;
}

bool ProfilePage::validatePage()
{
    auto *button = printerGroup->checkedButton();
    if (!button) {
        return false;
    }

    QString printerId = button->property("printerId").toString();

    // Save to AppConfig here
    emit printerTypeSelected(printerId);

    return true;
}

void ProfilePage::setAvailablePrinters(const QList<PrinterViewData> &printers)
{
    int id = 0;
    for (const auto& p : printers) {
        QRadioButton* btn = new QRadioButton(p.name, this);
        btn->setProperty("printerId", p.id);
        printerGroup->addButton(btn, id++);
        ui->printerGroupFrame->layout()->addWidget(btn); // Or some container layout
    }

    // DELETE THIS LATER!!!!! (doesn't show Cylinder 2 for hardcoded integration purpose)
    printerGroup->button(1)->setEnabled(false);
}

int ProfilePage::getSelectedPrinterType() const
{
    //return printerGroup->checkedButton()->property("printerId");
    return printerGroup->checkedId();
}

