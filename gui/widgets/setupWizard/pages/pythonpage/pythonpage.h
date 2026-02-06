#ifndef PYTHONPAGE_H
#define PYTHONPAGE_H

#include <QWizardPage>

namespace Ui {
class PythonPage;
}

class PythonPage : public QWizardPage
{
    Q_OBJECT

public:
    explicit PythonPage(QWidget *parent = nullptr);
    ~PythonPage();

    bool isComplete() const override;

signals:
    void pythonPathSelected(QString pythonPath);

private:
    Ui::PythonPage *ui;

private:
    void browseButtonClicked();
    bool isValidPython(const QString &path);
};

#endif // PYTHONPAGE_H
