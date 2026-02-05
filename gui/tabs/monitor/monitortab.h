#ifndef MONITORTAB_H
#define MONITORTAB_H

#include <QWidget>

namespace Ui {
class MonitorTab;
}

class MonitorTab : public QWidget
{
    Q_OBJECT

public:
    explicit MonitorTab(QWidget *parent = nullptr);
    ~MonitorTab();

private:
    Ui::MonitorTab *ui;
};

#endif // MONITORTAB_H
