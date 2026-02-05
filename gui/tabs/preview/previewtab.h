#ifndef PREVIEWTAB_H
#define PREVIEWTAB_H

#include <QWidget>

namespace Ui {
class PreviewTab;
}

class PreviewTab : public QWidget
{
    Q_OBJECT

public:
    explicit PreviewTab(QWidget *parent = nullptr);
    ~PreviewTab();

private:
    Ui::PreviewTab *ui;
};

#endif // PREVIEWTAB_H
