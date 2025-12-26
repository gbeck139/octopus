#include "previewtab.h"
#include "ui_previewtab.h"

PreviewTab::PreviewTab(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::PreviewTab)
{
    ui->setupUi(this);
}

PreviewTab::~PreviewTab()
{
    delete ui;
}
