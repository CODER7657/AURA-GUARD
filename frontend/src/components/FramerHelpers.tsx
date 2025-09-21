import React from 'react';
import { motion } from 'framer-motion';
import type { FramerComponentProps } from '@/types';

// Wrapper component for Framer exports
interface FramerWrapperProps extends FramerComponentProps {
  children: React.ReactNode;
  animationProps?: any;
}

export const FramerWrapper: React.FC<FramerWrapperProps> = ({
  children,
  animationProps = {},
  className = '',
  ...props
}) => {
  const defaultAnimation = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.3 }
  };

  return (
    <motion.div
      className={className}
      {...defaultAnimation}
      {...animationProps}
      {...props}
    >
      {children}
    </motion.div>
  );
};

// Higher-order component for enhancing Framer exports with API integration
export function withApiIntegration<P extends object>(
  FramerComponent: React.ComponentType<P>
) {
  return function EnhancedFramerComponent(props: P & FramerComponentProps) {
    const { onSubmit, onClick, isLoading, data, ...frameProps } = props;

    const handleInteraction = (interactionData?: any) => {
      if (onSubmit && interactionData) {
        onSubmit(interactionData);
      } else if (onClick) {
        onClick();
      }
    };

    return (
      <FramerWrapper isLoading={isLoading}>
        <FramerComponent
          {...(frameProps as P)}
          onSubmit={handleInteraction}
          onClick={handleInteraction}
          data={data}
        />
      </FramerWrapper>
    );
  };
}

// Example loading component with Framer Motion
export const LoadingSpinner: React.FC<{ size?: number }> = ({ size = 40 }) => {
  return (
    <motion.div
      className="flex items-center justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="border-4 border-blue-200 border-t-blue-600 rounded-full"
        style={{ width: size, height: size }}
        animate={{ rotate: 360 }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "linear"
        }}
      />
    </motion.div>
  );
};

// Example error component
export const ErrorMessage: React.FC<{ message: string; onRetry?: () => void }> = ({
  message,
  onRetry
}) => {
  return (
    <motion.div
      className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
    >
      <p className="font-medium">Error</p>
      <p className="text-sm mt-1">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700 transition-colors"
        >
          Try Again
        </button>
      )}
    </motion.div>
  );
};