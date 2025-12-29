#ifndef PROFILEPAGE_H
#define PROFILEPAGE_H

#include <QWizardPage>
#include <QButtonGroup>

namespace Ui {
class ProfilePage;
}

class ProfilePage : public QWizardPage
{
    Q_OBJECT

public:
    explicit ProfilePage(QWidget *parent = nullptr);
    ~ProfilePage();

    bool isComplete() const override;
    bool validatePage() override;

signals:
    void printerTypeSelected(QString printerId); // change int to PrinterType later? (enum)

private:
    Ui::ProfilePage *ui;
    QButtonGroup* printerGroup;

    int getSelectedPrinterType() const;
};

#endif // PROFILEPAGE_H
