#include "welcomepage.h"
#include "ui_welcomepage.h"

WelcomePage::WelcomePage(QWidget *parent)
    : QWizardPage(parent)
    , ui(new Ui::WelcomePage)
{
    ui->setupUi(this);

    setTitle("Welcome");
    setSubTitle("Initial application setup");
}

WelcomePage::~WelcomePage()
{
    delete ui;
}
