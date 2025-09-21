const { DataTypes } = require('sequelize');

const Notification = (sequelize) => {
  const NotificationModel = sequelize.define('Notification', {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    userId: {
      type: DataTypes.UUID,
      allowNull: true,
      references: {
        model: 'users',
        key: 'id',
      },
    },
    email: {
      type: DataTypes.STRING(255),
      allowNull: false,
      validate: {
        isEmail: true,
        notEmpty: true,
      },
    },
    type: {
      type: DataTypes.ENUM('air_quality_alert', 'forecast_alert', 'system_notification'),
      allowNull: false,
    },
    title: {
      type: DataTypes.STRING(255),
      allowNull: false,
      validate: {
        len: [1, 255],
        notEmpty: true,
      },
    },
    message: {
      type: DataTypes.TEXT,
      allowNull: false,
      validate: {
        notEmpty: true,
      },
    },
    severity: {
      type: DataTypes.ENUM('low', 'medium', 'high', 'critical'),
      defaultValue: 'medium',
      allowNull: false,
    },
    status: {
      type: DataTypes.ENUM('pending', 'sent', 'failed', 'cancelled'),
      defaultValue: 'pending',
      allowNull: false,
    },
    deliveryMethod: {
      type: DataTypes.ENUM('email', 'push', 'sms'),
      defaultValue: 'email',
      allowNull: false,
    },
    scheduledAt: {
      type: DataTypes.DATE,
      allowNull: true,
    },
    sentAt: {
      type: DataTypes.DATE,
      allowNull: true,
    },
    location: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    airQualityData: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    metadata: {
      type: DataTypes.JSONB,
      defaultValue: {},
      allowNull: true,
    },
  }, {
    tableName: 'notifications',
    timestamps: true,
    indexes: [
      { fields: ['user_id'] },
      { fields: ['email'] },
      { fields: ['type'] },
      { fields: ['status'] },
      { fields: ['severity'] },
      { fields: ['scheduled_at'] },
    ],
  });

  // Instance methods
  NotificationModel.prototype.markAsSent = function() {
    this.status = 'sent';
    this.sentAt = new Date();
    return this.save();
  };

  NotificationModel.prototype.markAsFailed = function() {
    this.status = 'failed';
    return this.save();
  };

  // Class methods
  NotificationModel.findPendingNotifications = function() {
    return this.findAll({
      where: { status: 'pending' },
      order: [['scheduledAt', 'ASC'], ['createdAt', 'ASC']],
    });
  };

  NotificationModel.findByUser = function(userId) {
    return this.findAll({
      where: { userId },
      order: [['createdAt', 'DESC']],
    });
  };

  NotificationModel.findByEmail = function(email) {
    return this.findAll({
      where: { email },
      order: [['createdAt', 'DESC']],
    });
  };

  return NotificationModel;
};

module.exports = Notification;