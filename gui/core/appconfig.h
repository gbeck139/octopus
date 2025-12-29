#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <QObject>
#include <QSettings>

///
/// \brief The AppConfig class stores user preferences to
/// persist settings across launches.
///
class AppConfig : public QObject
{
    Q_OBJECT
public:
    explicit AppConfig(QObject *parent = nullptr);

    // First-run flag
    bool isFirstRun() const;

    // Profile Settings
    void setActivePrinter(QString printerId);
    void setActiveMaterial(QString materialId);
    void setActiveProcess(QString processId);

    QString getActivePrinter() const;
    QString getActiveMaterial() const;
    QString getActiveProcess() const;

    //User Preferences

signals:

public slots:
    void setFirstRunCompleted(bool completed);

private:
    QSettings appSettings;
};

#endif // APPCONFIG_H
