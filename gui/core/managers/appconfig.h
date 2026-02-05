#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <QObject>
#include <QSettings>

///
/// \brief Stores persistent application-level user configuration
/// to persist settings across launches.
///
class AppConfig : public QObject
{
    Q_OBJECT
public:
    explicit AppConfig(QObject *parent = nullptr);

    // First-run
    bool isFirstRun() const;
    void markFirstRunCompleted();

    // Active profile IDs
    QString getActivePrinterId() const;
    QString getActiveMaterialId() const;
    QString getActiveProcessId() const;

    void setActivePrinterId(const QString& printerId);
    void setActiveMaterialId(const QString& materialId);
    void setActiveProcessId(const QString& processId);

    // TODO: Add more User Preferences here

signals:
    void activePrinterChanged(const QString& printerId);
    void activeMaterialChanged(const QString& materialId); /////////////////////
    void activeProcessChanged(const QString& processId); /////////////////////

//public slots:
    //void setFirstRunCompleted(bool completed);  ///////////////////// DELETED

private:
    QSettings appSettings;
};

#endif // APPCONFIG_H
