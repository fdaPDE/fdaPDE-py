#ifndef __FE_LS_FISHER_KPP_H__
#define __FE_LS_FISHER_KPP_H__

#include <fdaPDE/src/solvers/header_check.h>

namespace fdapde{
namespace internals{

template <typename FeSpace>
class fe_ls_fisher_kpp{
 private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using diag_matrix_t   = Eigen::DiagonalMatrix<double, Dynamic, Dynamic>;
    using sparse_solver_t = eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
    using dense_solver_t  = Eigen::PartialPivLU<matrix_t>;
    
    using Triangulation = FeSpace::Triangulation;
    
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;    
    
    using diffusion_t = FeCoeff<local_dim, local_dim, local_dim, matrix_t>;
    using reaction_t = FeCoeff<local_dim, 1, 1, vector_t>;
    using FeType = FeSpace::FeType;
   
    template <typename DataLocs>
    static constexpr bool is_valid_data_locs_descriptor_v =
      std::is_same_v<DataLocs, matrix_t> || std::is_same_v<DataLocs, binary_t>;
    
    class block_map_t {
        static constexpr int Order = 3;
        using Scalar = double;
        using storage_t = MdMap<Scalar, full_dynamic_extent_t<Order>, internals::layout_left>;

        storage_t data_;
        int rows_ = 0, cols_ = 0;
        int blk_rows_ = 0, blk_cols_ = 0;   // single block size
       public:
        block_map_t() noexcept = default;
        template <typename DataT>
            requires(internals::is_eigen_dense_xpr_v<DataT> &&
                     std::is_same_v<typename std::decay_t<DataT>::Scalar, double>)
        block_map_t(DataT&& data, int rows, int blk_rows, int blk_cols) :
            data_(data.data(), rows, data.cols(), (data.rows() / rows)),
            rows_(rows),
            cols_(data.cols()),
            blk_rows_(blk_rows),
            blk_cols_(blk_cols) {
            fdapde_assert(data.rows() % rows == 0 && rows % blk_rows == 0 && data.cols() % blk_cols == 0);
        }
        template <typename DataT>
            requires(internals::is_eigen_dense_xpr_v<DataT> &&
                     std::is_same_v<typename std::decay_t<DataT>::Scalar, double>)
        block_map_t(DataT&& data, int rows) :   // divide data in (data.rows() / rows) blocks of size rows x data.cols()
            data_(data.data(), rows, data.cols(), (data.rows() / rows)),
            rows_(rows),
	    cols_(data.cols()),
            blk_rows_(rows),
            blk_cols_(data.cols()) {
            fdapde_assert(data.rows() % rows == 0);
        }
        block_map_t(const block_map_t& other) :
            rows_(other.rows_), cols_(other.cols_), blk_rows_(other.blk_rows_), blk_cols_(other.blk_cols_) {
            for (size_t i = 0; i < data_.size(); ++i) { data_.data()[i] = other.data_.data()[i]; }
        }
        block_map_t& operator=(const block_map_t& other) {
            for (size_t i = 0; i < data_.size(); ++i) { data_.data()[i] = other.data_.data()[i]; }
            rows_ = other.rows_;
            cols_ = other.cols_;
            blk_rows_ = other.blk_rows_;
            blk_cols_ = other.blk_cols_;
            return *this;
        }
        // observers
        auto operator()(int i, int k) const {   // get i-th row-block of k-th time instant
            auto slice_ = data_.template slice<2>(k);
            return slice_.as_eigen_map().block(i * blk_rows_, 0, blk_rows_, cols_);
        }
        auto topRows(int j, int k) const {   // get the first j row-blocks of the k-th time instant
            return data_.template slice<2>(k).as_eigen_map().block(0, 0, j * blk_rows_, cols_);
        }
        auto operator()(int k) const { return data_.template slice<2>(k).as_eigen_map(); }
        int size() const { return data_.size(); }
        // modifiers
        auto operator()(int i, int k) {
            auto slice_ = data_.template slice<2>(k);
            return slice_.as_eigen_map().block(i * blk_rows_, 0, blk_rows_, cols_);
        }
        auto topRows(int j, int k) {
            return data_.template slice<2>(k).as_eigen_map().block(0, 0, j * blk_rows_, cols_);
        }
        auto operator()(int k) { return data_.template slice<2>(k).as_eigen_map(); }
    };

    template <typename DataLocs>
        requires(is_valid_data_locs_descriptor_v<DataLocs>)
    void eval_spatial_basis_at_(const DataLocs& locs) {
        if constexpr (std::is_same_v<DataLocs, matrix_t>) {   // pointwise sampling
            Psi_ = point_eval_(locs);
            D_ = vector_t::Ones(n_locs_).asDiagonal();
        } else {   // areal sampling
            const auto& [psi, measure_vect] = areal_eval_(locs);
            Psi_ = psi;
            D_ = measure_vect.asDiagonal();
        }
        fdapde_assert(n_locs_ == Psi_.rows());
        return;
    }
    template <typename GeoFrame> void eval_spatial_basis_at_(const GeoFrame& gf) {
        switch (gf.category(0)[0]) {
        case ltype::point: {
            const auto& spatial_index = geo_index_cast<0, POINT>(gf[0]);
            n_ = spatial_index.rows();
            if (spatial_index.points_at_dofs()) {
                Psi_.resize(n_, n_);
                Psi_.setIdentity();
            } else {
                Psi_ = point_eval_(spatial_index.coordinates());
            }
            D_ = vector_t::Ones(n_).asDiagonal();
            break;
        }
        case ltype::areal: {
            const auto& spatial_index = geo_index_cast<0, POLYGON>(gf[0]);
            n_ = spatial_index.rows();
            const auto& [psi, measure_vect] = areal_eval_(spatial_index.incidence_matrix());
            Psi_ = psi;
            D_ = measure_vect.asDiagonal();
            break;
        }
        }
        return;
    }
   
   public:
    static constexpr int n_lambda = 1;
    using solver_category = ls_solver;
    
    fe_ls_fisher_kpp() noexcept = default;

    template <typename GeoFrame, typename PDEparam, typename WeightMatrix>
    fe_ls_fisher_kpp(const std::string& formula, const GeoFrame& gf, PDEparam&& pde_param, const WeightMatrix& W):fe_space_(pde_param.fe_space()) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_TWO_GEOFRAMES_ONLY);
	fdapde_assert(gf.n_layers() == 1);
        n_obs_  = gf[0].rows();
        n_locs_ = n_obs_;
        std::cout << "0 :" << pde_param.fe_space().n_dofs() << std::endl;
        std::cout << "constr n_dofs " << fe_space_.n_dofs() << std::endl;
        discretize(pde_param);
        analyze_data(formula, gf, W);
    }
    template <typename GeoFrame, typename PDEparam>
    fe_ls_fisher_kpp(const std::string& formula, const GeoFrame& gf, PDEparam&& pde_param) :
        fe_ls_fisher_kpp(formula, gf, pde_param, vector_t::Ones(gf[0].rows()).asDiagonal()) { }
    // construct with no data
    template <typename GeoFrame, typename PDEparam, typename WeightMatrix>
    fe_ls_fisher_kpp(const GeoFrame& gf, PDEparam&& pde_param, const WeightMatrix& W) : fe_space_(pde_param.fe_space()), W_(W) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_TWO_GEOFRAMES_ONLY);
	fdapde_assert(gf.n_layers() == 1);
	n_obs_  = gf[0].rows();
	n_locs_ = n_obs_;

	// extract temporal mesh
        const auto& time_index = geo_index_cast<1, POINT>(gf[0]);
        const auto& time_coords = time_index.coordinates();
        m_ = time_coords.rows();
        fdapde_assert(m_ > 0 && time_coords.cols() == 1);
        DeltaT_ = time_coords(1, 0) - time_coords(0, 0);
        for (int i = 1; i < m_ - 1; ++i) {
            double lag_i = time_coords(i + 1, 0) - time_coords(i, 0);
            fdapde_assert(DeltaT_ > 0 && lag_i > 0 && almost_equal(DeltaT_ FDAPDE_COMMA lag_i));
        }

        discretize(pde_param);
        u_.resize(n_dofs_ * m_);
        for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
        eval_spatial_basis_at_(gf);
    }
    template <typename GeoFrame, typename PDEparam>
    fe_ls_fisher_kpp(const GeoFrame& gf, PDEparam&& pde_param) :
        fe_ls_fisher_kpp(gf, pde_param, vector_t::Ones(gf[0].rows()).asDiagonal()) { }

    template <typename PDEparam> 
    void discretize(PDEparam&& pde_param) {

        TrialFunction uh(fe_space_);
        TestFunction vh(fe_space_);
        
        diffusion_ = pde_param.diffusion();
        reaction_ = pde_param.reaction();

        auto bilinear_form = integral(fe_space_.triangulation())(dot(diffusion_*grad(uh), grad(vh)) - reaction_*uh*vh);
        
        // discretization
        n_dofs_ = bilinear_form.n_dofs();
        
        f_old = vector_t::Zero(n_dofs_);
        
        internals::fe_mass_assembly_loop<FeSpace> mass_assembler(bilinear_form.trial_space());
        M_ = mass_assembler.assemble();
        A_ = bilinear_form.assemble();
	    //u_space_ = linear_form.assemble();
	// store handles for basis system evaluation at locations
        point_eval_ = [fe_space = bilinear_form.trial_space()](const matrix_t& locs) -> decltype(auto) {
            return internals::point_basis_eval(fe_space, locs);
        };
        areal_eval_ = [fe_space = bilinear_form.trial_space()](const binary_t& locs) -> decltype(auto) {
            return internals::areal_basis_eval(fe_space, locs);
        };
	    
        // store initial conditions, numerical scheme parameters
        s_ = pde_param.ic();
        max_iter_ = pde_param.max_iter();
	    tol_ = pde_param.tol();

	    return;
    }
    
    // fit from formula
    template <typename GeoFrame, typename WeightMatrix>
    void analyze_data(const std::string& formula, const GeoFrame& gf, const WeightMatrix& W) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_ONE_GEOFRAMES_ONLY);
        fdapde_assert(gf.n_layers() == 1 && gf[0].category()[1] == ltype::point);
        n_obs_ = gf[0].rows();
        n_locs_ = n_obs_;

        // extract time step and number of time instants
        const auto& time_index = geo_index_cast<1, POINT>(gf[0]);
        const auto& time_coords = time_index.coordinates();
        m_ = time_coords.rows();
        fdapde_assert(m_ > 0 && time_coords.cols() == 1);
        DeltaT_ = time_coords(1, 0) - time_coords(0, 0);
        for (int i = 1; i < m_ - 1; ++i) {
            double lag_i = time_coords(i + 1, 0) - time_coords(i, 0);
            fdapde_assert(DeltaT_ > 0 && lag_i > 0 && almost_equal(DeltaT_ FDAPDE_COMMA lag_i));
        }
        if (u_.size() == 0) {
            // u_.resize(n_dofs_ * m_);
            u_ = vector_t::Zero(n_dofs_ * m_);
            //for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
        }

        // update forcing
	    u0_ = u_ + (1.0 / DeltaT_) * (M_ * s_);
        eval_spatial_basis_at_(gf);   // update \Psi matrix
        // parse formula, extract response vector
        Formula formula_(formula);
        const auto& y_data = gf[0].data().template col<double>(formula_.lhs());
        y_.resize(n_locs_, y_data.blk_sz());
        y_data.assign_to(y_);
	
	    update_response_and_weights(y_, W);
    }

    // modifiers
    void update_response(const vector_t& y) {
        fdapde_assert(Psi_.rows() > 0 && y.rows() == n_locs_ && y.cols() == 1);
        y_ = y;
	// correct \Psi for missing observations
        auto nan_pattern = na_matrix(y);
        if (nan_pattern.any()) {
            n_obs_ = n_locs_ - nan_pattern.count();
            B_.resize(m_);
            for (int i = 0; i < m_; ++i) {
                B_[i] = (~nan_pattern.middleRows(i * n_, n_)).repeat(1, n_dofs_).select(Psi_, 0);
            }
            y_ = (~nan_pattern).select(y_, 0);
        }	
    }
    template <typename WeightMatrix> void update_weights(const WeightMatrix& W) {
        fdapde_assert(Psi_.rows() > 0 && W.rows() == n_locs_ && W.rows() == W.cols());
        W_ = W;
        // check if W_ is time-wise block-constant
        W_const_ = true;
        for (int i = 1; i < m_; ++i) {
            if (sparse_matrix_t(W_.block(i * n_, i * n_, n_, n_) - W_.block(0, 0, n_, n_)).sum() != 0) {
                W_const_ = false;
                break;
            }
        }
        W_ /= n_obs_;
        W_changed_ = true;
        return;
    }
    template <typename WeightMatrix> void update_response_and_weights(const vector_t& y, const WeightMatrix& W) {
        update_response(y);
        update_weights (W);
        return;
    }

    // J(f,g) = \sum_{k=1}^m (y^k - \Psi*f^k)^T*(y^k - \Psi*f^k) + \lambda_S*(g^k)^T*(g^k) + \lambda_T*(l^k)^T*(l^k)
    double J(const vector_t& f__, const vector_t& g__, double lambda_D) const {
        block_map_t y(y_, n_);
        block_map_t f(f__, n_dofs_); 
        block_map_t g(g__, n_dofs_);
        double sse = 0;
        for (int t = 0; t < m_; ++t) {
            sse += (y(t) - Psi_ * f(t)).squaredNorm() + lambda_D * g(t).squaredNorm();
        }
        return sse;
    }

    struct obj_t{
        obj_t(fe_ls_fisher_kpp& model, double lambda) : model_(std::addressof(model)), lambda_(lambda) { }
        obj_t(fe_ls_fisher_kpp& model, double lambda, double tol) : model_(std::addressof(model)), lambda_(lambda), tol_(tol) { }

        double operator()(const vector_t& g__) const {
            vector_t f__ = model_->state(g__);
            block_map_t f(f__, model_->n_dofs()); 
            block_map_t g(g__, model_->n_dofs());
            block_map_t y(model_->response(), model_->n());
            double sse = 0;
            for (int t = 0; t < model_->m(); ++t) {
                sse += (y(t) - model_->Psi() * f(t)).squaredNorm() + lambda_ * g(t).squaredNorm();
            }
            return sse;
        }

        // gradient() deve restituire una lambda !
        std::function< vector_t(const vector_t&) > gradient() {
            return [this](const vector_t &g__) {
                auto f__ = model_->state(g__);
                auto p__ = model_->adjoint(f__);

                block_map_t g(g__, model_->n_dofs());
                block_map_t p(p__, model_->n_dofs());
                auto grad_t = vector_t::Zero(m_ * n_dofs_);
                for(int t = 0; t < model_->m(); ++t){
                    grad = lambda_ * g(t) - p(t);
                }
                return grad;
            };
        }

        template <typename Optimizer> bool stop_if(Optimizer &opt) {
            bool stop;
            double loss_old = this->operator()(opt.x_old);
            double loss_new = this->operator()(opt.x_new);
            stop = std::abs((loss_new - loss_old) / loss_old) < tol_;
            return stop;
        }

        private:
          fe_ls_fisher_kpp* model_;
          double lambda_;
          double tol_ = 1e-5;
    };
  
   private:
    
    vector_t state(const vector_t &g__) const {
        
        TrialFunction uh(fe_space_);
        TestFunction vh(fe_space_);

        auto reac = integral(fe_space_.triangulation())(reaction_ * f_old * uh * vh);

        vector_t f__ = vector_t::Zero(n_dofs_ * m_);
        f__.block(0, 0, n_dofs_, 1) = s_;

        block_map_t f(f__, n_dofs_);
        block_map_t g(g__, n_dofs_);
            
        for (int t = 0; t < m_ - 1; t++) {
            f_old = f(t);
            auto R_ = reac.assemble();
            R_.makeCompressed();

            sparse_matrix_t S = 1. / DeltaT_ * M_ + A_ + R_;
            S.makeCompressed();

            sparse_solver_t lin_solver(S);
            lin_solver.factorize(S);
            vector_t b = 1. / DeltaT_ * M_ * f_old.coeff() + M_ * g(t + 1);
            f(t + 1) = lin_solver.solve(b);
        }

        return f__;
    }

    vector_t adjoint(const vector_t &f__) const {

        TrialFunction uh(fe_space_);
        TestFunction vh(fe_space_);
        auto reac = integral(fe_space_.triangulation())(reaction_ * f_old * uh * vh);
     
        vector_t p__ = vector_t::Zero(n_dofs_ * m_);
        block_map_t p(p__, n_dofs_);
        block_map_t f(f__, n_dofs_);
        block_map_t y(y_, n_);
        
        int n_fact = (B_.size() != 0 || !W_const_) ? m_ : 1;   // != 1 if missing or heteroschedastic observations
        auto PsiNA = [&](int t) -> const sparse_matrix_t& { return B_.size() != 0 ? B_[t] : Psi_; };
        auto W = [&](int i) { return W_.block(i * n_, i * n_, n_, n_); };
        
        for (int t = m_ - 1; t > 0; t--) {
            f_old = f(t - 1);
            auto R_ = reac.assemble();
            R_.makeCompressed();

            sparse_matrix_t S = 1. / DeltaT_ * M_ + A_ + 2. * R_;
            S.makeCompressed();

            sparse_solver_t lin_solver(S);
            lin_solver.factorize(S);

            vector_t b =
                1. / DeltaT_ * M_ * p(t) -
                //(1 - lambda_) * PsiTPsi_ * get_t(y, t - 1) +
                //(1 - lambda_) * Psi_.transpose() * obs(t - 1);
                1. / (m_ * n_) * PsiNA(t).transpose() * D_ * W(t) * PsiNA(t) * f(t-1) +
                1. / (m_ * n_) * PsiNA(m_ - 1).transpose() * D_ * W(m_ - 1) * y(t - 1);

            p(t - 1) = lin_solver.solve(b);
    }

    return p__;
  }

   public:
    // main fit entry point
    template <typename Optimizer, typename... Callbacks>
    const vector_t& fit(double lambda, const vector_t& g_init, Optimizer&& opt, Callbacks&&... callbacks) {
        
        g_ = opt.optimize(obj_t(*this, lambda, tol_), g_init, std::forward<Callbacks>(callbacks)...);
        f_ = state(g_);
        p_ = adjoint(f_);
        return f_;
    }

    template <typename LambdaT>
        requires(internals::is_vector_like_v<LambdaT>)
    const vector_t& fit(LambdaT&& lambda) {
        fdapde_assert(lambda.size() == n_lambda);
        return fit(lambda[0]);
    }

    // observers
    int n_dofs() const { return n_dofs_; }
    int n() const { return n_; }
    int m() const { return m_; }
    const sparse_matrix_t& mass() const { return M_; }
    const sparse_matrix_t& stiff() const { return A_; }
    const sparse_matrix_t& Psi() const { return Psi_; }
    const vector_t& force() const { return u_; }
    const vector_t& f() const { return f_; }
    const vector_t& beta() const { return beta_; }
    const vector_t& misfit() const { return g_; }
    const vector_t& response() const { return y_; }
   protected:
    std::optional<std::array<double, n_lambda>> lambda_saved_ = std::array<double, n_lambda> {-1};
    sparse_solver_t invA_, invAs_;
    
    int n_dofs_ = 0, n_obs_ = 0, n_covs_ = 0;
    int n_locs_ = 0, n_ = 0, m_ = 0;   // n_: number of spatial locations, m_: number of time instants

    sparse_matrix_t M_;    // n_dofs x n_dofs matrix [R0]_{ij} = \int_D \psi_i * \psi_j
    sparse_matrix_t A_;    // n_dofs x n_dofs matrix [R1]_{ij} = \int_D a(\psi_i, \psi_j)
    sparse_matrix_t R_;     // n_dofs x n_dofs matrix [R]_{ij} = \int_D (r * \f \psi_i * \psi_j) // non linear reaction
    sparse_matrix_t Psi_;   // n_obs x n_dofs matrix [Psi]_{ij} = \psi_j(p_i)
    std::vector<sparse_matrix_t> B_;   // m x (n_obs x n_obs) vector of na-corrected \Psi matrices
    vector_t u_, u0_;            // (n_dofs * m) x 1 vector u = [u_1 + + M_*s / DeltaT, u_2, \ldots, u_n]
    vector_t u_space_;
    vector_t s_;             // initial condition vector
    diag_matrix_t D_;       // vector of regions' measures (areal sampling)
    mutable sparse_solver_t invR0_;
    vector_t f_, beta_, g_, p_;

    // basis system evaluation handles
    std::function<sparse_matrix_t(const matrix_t& locs)> point_eval_;
    std::function<std::pair<sparse_matrix_t, vector_t>(const binary_t& locs)> areal_eval_;

    vector_t y_;          // n_obs x 1 observation vector
    sparse_matrix_t W_;   // n_obs x n_obs matrix of observation weights
    bool W_changed_, W_const_;   // W_const_ == true \iff W_ is time-wise constant

    int max_iter_;   // maximum number of iterations
    double tol_;     // convergence tolerance
    double DeltaT_;
    
    FeSpace fe_space_;
    FeFunction<FeSpace> f_old;

    diffusion_t diffusion_;
    reaction_t reaction_;  
};

} // internals


template <typename PDEparam, typename FeSpace>
struct fe_ls_fisher_kpp {

    using Scalar = double;
    using solver_t = internals::fe_ls_fisher_kpp<FeSpace>;
    using Triangulation = FeSpace::Triangulation;
    using matrix_t = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<Scalar, Dynamic, 1>;

    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;    
    
    using diffusion_t = FeCoeff<local_dim, local_dim, local_dim, matrix_t>;
    //using advection_t = FeCoeff<local_dim, local_dim, 1, matrix_t>;
    using reaction_t = FeCoeff<local_dim, 1, 1, vector_t>;

    using FeType = FeSpace::FeType;
    using Quadrature = FeType::template cell_quadrature_t<local_dim>;
    //static constexpr int n_quadrature_nodes = Quadrature::order;

   private:
    struct pde_param_packet {
        using Diffusion = std::tuple_element_t<0, std::decay_t<PDEparam>>;
        using Reaction = std::tuple_element_t<1, std::decay_t<PDEparam>>;
       private:

        matrix_t diffusion_mtx;
        vector_t reaction_mtx;

        diffusion_t diffusion_;
        reaction_t reaction_;
        FeSpace& fe_space_;
        vector_t ic_;
        int max_iter_ = 100;
        double tol_ = 1e-4;
        int n_quadrature_nodes = 0;

       public:
        pde_param_packet(const Diffusion& diffusion, const Reaction& reaction, FeSpace& fe_space, const vector_t& ic, 
                         int max_iter, double tol) :
                fe_space_(fe_space), ic_(ic), max_iter_(max_iter), tol_(tol), 
                n_quadrature_nodes(Quadrature::order * fe_space.triangulation().n_cells()) {
                
                std::cout << "ciao" << std::endl;
                // diffusion
                matrix_t mu = matrix_t::Zero(local_dim, local_dim);
                if constexpr ( std::is_same_v<Diffusion, Scalar> ){
                    for(int i=0; i < local_dim; ++i) mu(i,i) = diffusion;
                    diffusion_mtx = mu.reshaped().transpose().replicate(n_quadrature_nodes,1);
                    diffusion_ = diffusion_t(diffusion_mtx);                    
                } else if constexpr ( std::is_same_v<Diffusion, Eigen::Matrix<Scalar, local_dim, local_dim>> ){
                    for(int i =0; i < local_dim; ++i){
                        for(int j=0; j < local_dim; ++j){
                            mu(i,j) = diffusion(i,j);
                        }
                    }
                    diffusion_mtx = mu.reshaped().transpose().replicate(n_quadrature_nodes,1);
                    diffusion_ = diffusion_t(diffusion_mtx);
                } else if constexpr( std::is_same_v<Diffusion, diffusion_t> ){
                    diffusion_mtx = diffusion.data();
                    diffusion_ = diffusion;
                }
                
                // reaction 
                if constexpr ( std::is_same_v<Reaction, Scalar> ){
                    vector_t reac = vector_t::Zero(1);
                    reac[0] = reaction;
                    reaction_mtx = reac.replicate(n_quadrature_nodes,1);
                    reaction_ = reaction_t(reaction_mtx);
                } else if constexpr( std::is_same_v<Reaction, reaction_t> ){
                    reaction_mtx = reaction.data();
                    reaction_ = reaction;
                }
        }
        // observers
        const diffusion_t& diffusion() const { return diffusion_; }
        const reaction_t& reaction() const { return reaction_; }
        FeSpace& fe_space() const { return fe_space_; }
        int max_iter() const { return max_iter_; }
        double tol() const { return tol_; }
        const vector_t& ic() const { return ic_;}
    };
   public:
    fe_ls_fisher_kpp(const PDEparam& pde_param, FeSpace& fe_space, vector_t ic, int max_iter = 100, double tol = 1e-4) :
        pde_param_(std::get<0>(pde_param), std::get<1>(pde_param), fe_space, ic, max_iter, tol) { }
    const pde_param_packet& get() const { return pde_param_; }
   private:
    pde_param_packet pde_param_;
};

} // fdapde

#endif // __FE_LS_FISHER_KPP_H__