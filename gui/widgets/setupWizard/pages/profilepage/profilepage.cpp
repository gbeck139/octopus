#include "profilepage.h"
#include "ui_profilepage.h"

ProfilePage::ProfilePage(QWidget *parent)
    : QWizardPage(parent)
    , ui(new Ui::ProfilePage)
{
    ui->setupUi(this);

    setTitle("Default Profiles");
    setSubTitle("Choose printer type");

    // May have to change later when there is more printers.
    // (Needs data structure that holds system/user Printer Profile Data)

    // Images
    QPixmap cylOnePix(":/images/images/blueCylinder.jpg");
    int widthOne = ui->cylinderOneImage->width();
    int hieghtOne = ui->cylinderOneImage->height();
    ui->cylinderOneImage->setPixmap(cylOnePix.scaled(widthOne,hieghtOne,Qt::KeepAspectRatio));

    QPixmap cylTwoPix(":/images/images/redCylinder.jpg");
    int widthTwo = ui->cylinderTwoImage->width();
    int heightTwo = ui->cylinderTwoImage->height();
    ui->cylinderTwoImage->setPixmap(cylTwoPix.scaled(widthTwo,heightTwo,Qt::KeepAspectRatio));

    printerGroup = new QButtonGroup(this);

    printerGroup->addButton(ui->cylinderOne, 1);
    printerGroup->addButton(ui->cylinderTwo, 2);
    printerGroup->addButton(ui->radioButton, 3);

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
    int printerType = printerGroup->checkedId();

    // Save to AppConfig here
    emit printerTypeSelected(printerType);

    return true;
}

int ProfilePage::getSelectedPrinterType() const
{
    return printerGroup->checkedId();
}

