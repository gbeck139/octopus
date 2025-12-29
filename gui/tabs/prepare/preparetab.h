#ifndef PREPARETAB_H
#define PREPARETAB_H

#include <QWidget>

namespace Ui {
class PrepareTab;
}

class PrepareTab : public QWidget
{
    Q_OBJECT

public:
    explicit PrepareTab(QWidget *parent = nullptr);
    ~PrepareTab();

public slots:
    void onPrinterChanged(const QString& printerId);

private:
    Ui::PrepareTab *ui;
};

#endif // PREPARETAB_H
