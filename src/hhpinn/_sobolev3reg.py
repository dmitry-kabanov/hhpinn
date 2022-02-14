r"""
Module contains class Sobolev3Reg that implements regularization of the third
derivative for a function R^2 \to R.
"""
import tensorflow as tf


def sobolev3reg(model):
    """Compute Sobolev H3 regularizer for the function `model`.

    Usage:

        s3r = sobolev3reg(model)
        result = s3r(pts)

    that is, s3r is the function that knows the model and, when invoked,
    evaluates the regularizer at points `pts`.

    Parameters
    ----------
    model : Keras model
        Must implement a mathematical function that maps R^2 to R.
    """
    @tf.function
    def compute(pts):
        with tf.GradientTape(
            persistent=True, watch_accessed_variables=False
        ) as t3:
            t3.watch(pts)
            with tf.GradientTape(
                persistent=True, watch_accessed_variables=False
            ) as t2:
                t2.watch(pts)
                with tf.GradientTape(
                    persistent=True, watch_accessed_variables=False
                ) as t1:
                    t1.watch(pts)
                    f = model(pts, training=True)

                # Compute velocity predictions from the stream function `psi`.
                # To simplify notation, we denote components of the gradient
                # of f to be (u, v).
                df = t1.gradient(f, pts)
                u, v = tf.split(df, 2, axis=1)

            grad_u = t2.gradient(u, pts)
            grad_v = t2.gradient(v, pts)

            du_dx, du_dy = tf.split(grad_u, 2, axis=1)
            dv_dx, dv_dy = tf.split(grad_v, 2, axis=1)

        grad_du_dx = t3.gradient(du_dx, pts)
        grad_du_dy = t3.gradient(du_dy, pts)
        grad_dv_dx = t3.gradient(dv_dx, pts)
        grad_dv_dy = t3.gradient(dv_dy, pts)

        d2u_dxx, d2u_dxy = tf.split(grad_du_dx, 2, axis=1)
        d2u_dyx, d2u_dyy = tf.split(grad_du_dy, 2, axis=1)
        d2v_dxx, d2v_dxy = tf.split(grad_dv_dx, 2, axis=1)
        d2v_dyx, d2v_dyy = tf.split(grad_dv_dy, 2, axis=1)

        reg_3 = (
            d2u_dxx ** 2
            + d2u_dxy ** 2
            + d2u_dyx ** 2
            + d2u_dyy ** 2
            + d2v_dxx ** 2
            + d2v_dxy ** 2
            + d2v_dyx ** 2
            + d2v_dyy ** 2
        )

        return reg_3

    return compute
