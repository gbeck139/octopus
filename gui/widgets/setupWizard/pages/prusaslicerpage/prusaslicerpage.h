#ifndef PRUSASLICERPAGE_H
#define PRUSASLICERPAGE_H

#include <QWizardPage>

namespace Ui {
class PrusaSlicerPage;
}

class PrusaSlicerPage : public QWizardPage
{
    Q_OBJECT

public:
    explicit PrusaSlicerPage(QWidget *parent = nullptr);
    ~PrusaSlicerPage();

    bool isComplete() const override;

private:
    Ui::PrusaSlicerPage *ui;

private:
    void browseButtonClicked();
    bool isValidPrusaSlicer(const QString &path);
};

#endif // PRUSASLICERPAGE_H
