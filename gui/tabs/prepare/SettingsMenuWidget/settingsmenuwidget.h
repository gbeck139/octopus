#ifndef SETTINGSMENUWIDGET_H
#define SETTINGSMENUWIDGET_H

#include <QWidget>

namespace Ui {
class SettingsMenuWidget;
}

class SettingsMenuWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SettingsMenuWidget(QWidget *parent = nullptr);
    ~SettingsMenuWidget();

private:
    Ui::SettingsMenuWidget *ui;
};

#endif // SETTINGSMENUWIDGET_H
