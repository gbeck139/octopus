#ifndef WELCOMEPAGE_H
#define WELCOMEPAGE_H

#include <QWizardPage>

namespace Ui {
class WelcomePage;
}

class WelcomePage : public QWizardPage
{
    Q_OBJECT

public:
    explicit WelcomePage(QWidget *parent = nullptr);
    ~WelcomePage();

private:
    Ui::WelcomePage *ui;
};

#endif // WELCOMEPAGE_H
